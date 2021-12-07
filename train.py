import datetime
import os
import os.path as osp
import time

import presets
import torch
import torch.utils.data
import torchvision
import utils
from coco_utils import get_coco
from torch import nn
import json 
import datetime 
import wandb
import warnings
warnings.filterwarnings(action='ignore') 

try:
    from torchvision.prototype import models as PM
except ImportError:
    PM = None

def get_dataset(dir_path, dataset_type, mode, transform, num_classes):
    paths = {
        "coco": (dir_path, get_coco, num_classes),
    }

    ds_path, ds_fn, num_classes = paths[dataset_type]
    ds = ds_fn(ds_path, mode=mode, transforms=transform, num_classes=num_classes)
    return ds, num_classes


# def get_transform(train, args):
#     if train:
#         return presets.SegmentationPresetTrain(base_size=520, crop_size=480)
#     elif not args.weights:
#         return presets.SegmentationPresetEval(base_size=520)
#     else:
#         fn = PM.segmentation.__dict__[args.model]
#         weights = PM._api.get_weight(fn, args.weights)
#         return weights.transforms()

def get_transform(train, base_size, crop_size):
    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)

def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


def evaluate(model, criterion, data_loader, device, num_classes):
    model.eval()
    val_loss = 0
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.inference_mode():
        for image, target, fn in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)

            loss = criterion(output, target)
            output = output["out"]

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat, loss


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, wandb=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for image, target, fn in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        # logging >>>
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        if wandb != None:
            wandb.log({"train_loss": loss.item(), "learning rate": optimizer.param_groups[0]["lr"]})


def main(args, weights_path):
    
    if args.weights and PM is None:
        raise ImportError("The prototype module couldn't be found. Please install the latest torchvision nightly.")
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)


    # dataset, num_classes = get_dataset(args.data_path, args.dataset, "train", get_transform(True, args), args.num_classes)
    # dataset_test, _ = get_dataset(args.data_path, args.dataset, "val", get_transform(False, args), args.num_classes)

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #     test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    # else:
    #     train_sampler = torch.utils.data.RandomSampler(dataset)
    #     test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     sampler=train_sampler,
    #     num_workers=args.workers,
    #     collate_fn=utils.collate_fn,
    #     drop_last=True,
    # )

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    # )

    dataset, num_classes = get_dataset(args.data_path, args.dataset_type, "train", get_transform(True, args.base_imgsz, args.crop_imgsz), args.num_classes)
    dataset_test, _ = get_dataset(args.data_path, args.dataset_type, "val", get_transform(False, args.base_imgsz, args.crop_imgsz), args.num_classes)

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.num_workers,
        collate_fn=utils.collate_fn, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

    if not args.weights:
        if args.model == 'deeplabv3_resnet101':
            model = torchvision.models.segmentation.__dict__[args.model](
                        pretrained=True,
                        aux_loss=args.aux_loss,
                        # num_classes=num_classes
                    )       
            model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)
            model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        else:
            model = torchvision.models.segmentation.__dict__[args.model](
                pretrained=args.pretrained,
                num_classes=num_classes,
                aux_loss=args.aux_loss,
            )
        print(">>> LODED TORCHVISION MODEL: ", args.model)
    
    else:
        model = PM.segmentation.__dict__[args.model](
            weights=args.weights, num_classes=num_classes, aux_loss=args.aux_loss
        )
        print(">>> LODED PM MODEL: ", args.model)
        print(PM)

    model.to(args.device)


    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # params_to_optimize = model.parameters()
    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    iters_per_epoch = len(data_loader)
    main_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: (1 - x / (iters_per_epoch * (args.epochs - args.lr_warmup_epochs))) ** 0.9
    )

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            print(">>>> RELOAD ALL !!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(">>>>>>> RELOAD MODEL", (not args.test_only))

    if len(args.device_ids) > 1: ## Important! Need to locate after parameter settings for optimization
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    if args.test_only:
        confmat = evaluate(model, data_loader_test, device=args.device, num_classes=num_classes)
        print(confmat)
        return

    ### For wandb
    if args.wandb:
        wandb.init(project=args.project_name, reinit=True)
        wandb.run.name = args.run_name
        wandb.config.update(args)
        wandb.watch(model)


    min_loss = 999
    info_weights = {'best': None, 'last': None}
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.wandb:
            train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, args.device, epoch, args.print_freq, wandb)
            confmat, loss_val = evaluate(model, criterion, data_loader_test, device=args.device, num_classes=num_classes)
            
            acc_global, acc, iu = confmat.compute()
            wandb.log({"val_loss": loss_val.item()})
            
            for class_name, val in zip(args.classes, (iu*100).tolist()):
                wandb.log({class_name + '_iou': val})
            wandb.log({'mean iou': iu.mean().item()*100})
            for class_name, val in zip(args.classes, (acc*100).tolist()):
                wandb.log({class_name + '_acc': val})
            
        else:
            train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, args.device, epoch, args.print_freq, None)
            confmat, loss_val = evaluate(model, criterion, data_loader_test, device=args.device, num_classes=num_classes, wandb=None)
        
        print(confmat)
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }

        if min_loss > loss_val:
            min_loss = loss_val
            utils.save_on_master(checkpoint, os.path.join(weights_path, "best.pth"))
            info_weights['best'] = epoch
            print(">>> Saved the best model .......! ")

        utils.save_on_master(checkpoint, os.path.join(weights_path, "last.pth"))
        info_weights['last'] = epoch

        with open(osp.join(weights_path, 'info.txt'), 'w') as f:
            f.write('best: {}'.format(info_weights['best']))
            f.write('\n')
            f.write('last: {}'.format(info_weights['last']))
            f.write('\n')    
        
    total_time = time.time() - start_time

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)
    
    parser.add_argument('--project-name', default='interojo')
    # parser.add_argument('--data-path', default='/home/wonchul/HDD/datasets/projects/interojo/3rd_poc_/coco_datasets_good/react_bubble_damage_print_dust')
    parser.add_argument('--data-path', default='/home/nvadmin/wonchul/mnt/HDD/datasets/projects/interojo/3rd_poc_/coco_datasets_good/react_bubble_damage_print_dust', help='dataset path')
    parser.add_argument('--wandb', default=True)
    parser.add_argument('--dataset-type', default='coco', help='dataset name')
    parser.add_argument("--model", default="deeplabv3_resnet101", type=str, help="model name")
    parser.add_argument("--aux-loss", action="store_true", help="auxiliar loss")
    parser.add_argument('--device', default='cuda', help='gpu device ids')
    parser.add_argument('--device-ids', default='2,3', help='gpu device ids')
    parser.add_argument("--batch-size", default=64, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--num-workers", default=32, type=int, metavar="N", help="number of data loading workers (default: 16)")
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("-weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument('--output-dir', default='./outputs/train', help='path where to save')
    # parser.add_argument("--resume", default="./outputs/train/1/weights/last.pth", type=str, help="path of checkpoint")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--pretrained", default=True)
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    # Prototype models only
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument('--base-imgsz', default=80, type=int, help='base image size')
    parser.add_argument('--crop-imgsz', default=80, type=int, help='base image size')

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    args.device_ids = list(map(int, args.device_ids.split(',')))
    args.classes = ['_background_'] + list(osp.split(osp.splitext(args.data_path)[0])[-1].split('_'))
    args.num_classes = len(list(osp.split(osp.splitext(args.data_path)[0])[-1].split('_'))) + 1
    if args.device == 'cuda':
        args.device = 'cuda:{}'.format(args.device_ids[0])

    now = datetime.datetime.now()
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            args.output_dir = os.path.join(args.output_dir, 'seg1')
            os.makedirs(args.output_dir)
        else:
            folders = os.listdir(args.output_dir)
            args.output_dir = os.path.join(args.output_dir, 'seg' + str(len(folders) + 1))
            os.makedirs(args.output_dir)

        output_path = args.output_dir
        args.date = str(datetime.datetime.now())
        utils.mkdir(osp.join(output_path, 'cfg'))
        weights_path = osp.join(output_path, 'weights')
        utils.mkdir(weights_path)

    args.run_name = osp.split(osp.splitext(output_path)[0])[-1]
    with open(osp.join(output_path, 'cfg/config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print(args)

    main(args, weights_path)
