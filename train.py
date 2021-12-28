import os
import os.path as osp
import sys 
sys.path.append(osp.join(osp.dirname(__file__), 'src'))
sys.path.append(osp.join(osp.dirname(__file__), 'models'))

import datetime
import time

import presets
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
import utils
from coco_utils import get_coco
from torch import nn
import json 
import datetime 
import wandb
import collections
import matplotlib.pyplot as plt 

### moduel 
from models.models import CreateModel

### distributed 
import torch.distributed as dist
import torch.multiprocessing as mp
from losses  import CELoss, DiceLoss
from pytorch_toolbelt import losses as L
from parallel import DataParallelModel, DataParallelCriterion

### ignore warnings
import warnings
warnings.filterwarnings(action='ignore') 

def get_dataset(dir_path, dataset_type, mode, transform, num_classes):
    paths = {
        "coco": (dir_path, get_coco, num_classes),
    }

    ds_path, ds_fn, num_classes = paths[dataset_type]
    ds = ds_fn(ds_path, mode=mode, transforms=transform, num_classes=num_classes)
    return ds, num_classes


def get_transform(train, base_size, crop_size):
    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)


def evaluate(model, criterion, data_loader, device, num_classes, deep_supervision, wandb=None):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.inference_mode():
        for image, target, fn in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)

            if deep_supervision:
                outputs = model(image)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
            else:
                output = model(image)
                loss = criterion(output, target)
            if isinstance(output, collections.OrderedDict) and 'out' in output.keys():
                output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())
        confmat.reduce_from_all_processes()

    return confmat, loss


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, deep_supervision, wandb=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for image, target, fn in metric_logger.log_every(data_loader, print_freq, header):
        # plt.imshow(image[0].permute(1, 2, 0).detach().numpy())
        # plt.imshow(target[0].unsqueeze(0).permute(1, 2, 0).detach().numpy(), alpha=0.8)
        # plt.show()
        image, target = image.to(device), target.to(device)

        
        if deep_supervision: 
            outputs = model(image)     
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
        else:
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


def main(args):

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size/len(args.device_ids))
        args.num_workers = int(args.num_workers/len(args.device_ids))


    ### SET DATALOADER ---------------------------------------------------------------------------------------------------
    dataset, num_classes = get_dataset(args.data_path, args.dataset_type, "train", get_transform(True, args.base_imgsz, args.crop_imgsz), args.num_classes)
    dataset_test, _ = get_dataset(args.data_path, args.dataset_type, "val", get_transform(False, args.base_imgsz, args.crop_imgsz), args.num_classes)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.num_workers,
        collate_fn=utils.collate_fn, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.num_workers,
        collate_fn=utils.collate_fn)


    ### SET MODEL -------------------------------------------------------------------------------------------------------
    model, params_to_optimize = CreateModel(args)
    print(">>> LODED TORCHVISION MODEL: ", args.model)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model

    # ### SET OPTIMIZER & SCHEDULER ------------------------------------------------------------------------------------------
    # params_to_optimize = [
    #     {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
    #     {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    # ]
    # if args.aux_loss:
    #     params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
    #     params_to_optimize.append({"params": params, "lr": args.lr * 10})

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum,
                              nesterov=args.nesterov, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    iters_per_epoch = len(data_loader_train)
    main_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: (1 - x / (iters_per_epoch * (args.epochs - args.lr_warmup_epochs))) ** 0.9
    )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.device_ids[0])
        model_without_ddp = model.module
        
    if len(args.device_ids) > 1 and args.dataparallel: ## Important! Need to locate after parameter settings for optimization
        model = torch.nn.DataParallel(model, device_ids=args.device_ids, output_device=args.device_ids[0])
        print(">>> The training is executed by dataparallel")

    model.to(args.device)

    ### SET LOSS FUNCTION -----------------------------------------------------------------------------------------------
    if args.loss == 'CE':
        criterion = CELoss(args.aux_loss)
    elif args.loss == 'DiceLoss':
        criterion = DiceLoss(num_classes, False)
    elif args.loss == 'BceDiceLoss':
        criterion = DiceLoss(num_classes, True)

    # if args.dataparallel:
    #     criterion = DataParallelCriterion(criterion)

    ### SET TRAIN PARAMETERS -------------------------------------------------------------------------------------------
    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            # warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            warmup_lr_scheduler = utils.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        elif args.lr_warmup_method == "constant":
            # warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
            warmup_lr_scheduler = utils.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        # lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        lr_scheduler = utils.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.weights:
        checkpoint = torch.load(args.weights, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        print(">>>> RELOAD ALL !!!!!!!!!!!!!!!!!!!!!!!!!!")


    ### SET WANDB ---------------------------------------------------------------------------------------------------------------------
    if args.wandb:
        wandb.init(project=args.project_name, reinit=True)
        if args.data_path.split('/')[-2].split('_')[-1] == 'good' or args.with_good:
            wandb.run.name = args.run_name + '_' + args.model + '_good'
        else:    
            wandb.run.name = args.run_name + '_' + args.model
        wandb.config.update(args)
        wandb.watch(model)


    ### START TRAINING --------------------------------------------------------------------------------------------------------------------
    print(">>> Start training .........................................")
    min_loss = 999
    info_weights = {'best': None, 'last': None}
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.wandb:
            train_one_epoch(model, criterion, optimizer, data_loader_train, lr_scheduler, args.device, epoch, 
                                                                    args.print_freq, args.deep_supervision, wandb)
            confmat, loss_val = evaluate(model, criterion, data_loader_test, device=args.device, num_classes=num_classes, 
                                                                                        deep_supervision=args.deep_supervision)
            
            acc_global, acc, iu = confmat.compute()
            wandb.log({"val_loss": loss_val.item()})
            
            for class_name, val in zip(args.classes, (iu*100).tolist()):
                wandb.log({class_name + '_iou': val})
            wandb.log({'mean iou': iu.mean().item()*100})
            for class_name, val in zip(args.classes, (acc*100).tolist()):
                wandb.log({class_name + '_acc': val})
            
        else:
            train_one_epoch(model, criterion, optimizer, data_loader_train, lr_scheduler, args.device, epoch, args.print_freq, 
                                                                                                                args.deep_supervision, None)
            confmat, loss_val = evaluate(model, criterion, data_loader_test, device=args.device, num_classes=num_classes, 
                                                                                                deep_supervision=args.deep_supervision, wandb=None)
        
        print(args.classes)
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
            utils.save_on_master(checkpoint, os.path.join(args.weights_path, "best.pth"))
            info_weights['best'] = epoch
            print(">>> Saved the best model .......! ")

        utils.save_on_master(checkpoint, os.path.join(args.weights_path, "last.pth"))
        info_weights['last'] = epoch

        with open(osp.join(args.weights_path, 'info.txt'), 'w') as f:
            f.write('best: {}'.format(info_weights['best']))
            f.write('\n')
            f.write('last: {}'.format(info_weights['last']))
            f.write('\n')    
        
    total_time = time.time() - start_time

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch based semantic segmentation", add_help=add_help)
    
    parser.add_argument('--project-name', default='INTEROJO_S_Factory')
    parser.add_argument('--data-path', default='/home/wonchul/HDD/datasets/projects/interojo/S_factory_ver2/coco_datasets_good/DUST', help='dataset path')
    # parser.add_argument('--data-path', default='/home/nvadmin/wonchul/mnt/HDD/datasets/projects/interojo/S_factory/coco_datasets_good/DUST_BUBBLE_DAMAGE_EDGE_RING_LINE_OVERLAP', help='dataset path')
    parser.add_argument('--with-good', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--dataset-type', default='coco', help='dataset name')

    # Model setting
    parser.add_argument("--model", default="NestedUNet", type=str, 
            help="For torchvision,  deeplabv3_resnet50 | deeplabv3_resnet101 | deeplabv3_mobilenet_v3_large | lraspp_mobilenet_v3_large" +
                 "For UNet**, NestedUNet | UNet")
    parser.add_argument("--pretrained", default=True)
    parser.add_argument("--weights", default=None, type=str, help="the weights to load")
    parser.add_argument("--input-channels", default=3)
    parser.add_argument('--base-imgsz', default=160, type=int, help='base image size')
    parser.add_argument('--crop-imgsz', default=160, type=int, help='base image size')

    # loss
    parser.add_argument("--loss", default='BceDiceLoss', help='CE | DiceLoss | BceDiceLoss')
    parser.add_argument("--aux-loss", action="store_true", help="auxiliar loss")
    
    # device
    parser.add_argument('--device', default='cuda', help='gpu device ids')
    parser.add_argument('--device-ids', default='0,1', help='gpu device ids')
    
    # training parameters
    parser.add_argument("--batch-size", default=1, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    
    parser.add_argument("--epochs", default=800, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument('--dataparallel', action='store_true')
    parser.add_argument("--num-workers", default=32, type=int, metavar="N", help="number of data loading workers (default: 16)")

    # optimizer & lr 
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--lr-warmup-epochs", default=30, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    
    parser.add_argument("--optimizer", default='SGD', help='SGD | Adam')
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument('--nesterov', default=False, help='nesterov')

    # if the model is unetpp
    parser.add_argument('--deep-supervision', action='store_true')

    # distributed training parameters
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default='tcp://127.0.0.1:5001', type=str, help="url used to set up distributed training")
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')

    # Prototype models only

    
    # etc.
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument('--output-dir', default='./outputs/train', help='path where to save')
    
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    if args.deep_supervision and (args.model != "NestedUNet" or args.model != "UNet"):
        raise RuntimeError("Deep supervision option can be applied with NestedUNet or UNet !!!!!!!")

    if args.model == "NestedUNet" or args.model == "UNet":
        if args.crop_imgsz/160 != 0:
            raise RuntimeError("UNet and NestedUNet should have image size multiplied by 160 !!!!!!!")

    args.device_ids = list(map(int, args.device_ids.split(',')))

    args.world_size = len(args.device_ids)*args.world_size
    args.rank = args.rank*len(args.device_ids) + args.device_ids[0]

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
        args.weights_path = osp.join(output_path, 'weights')
        utils.mkdir(args.weights_path)

    args.run_name = osp.split(osp.splitext(output_path)[0])[-1]
    with open(osp.join(output_path, 'cfg/config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    if args.output_dir:
        utils.mkdir(args.output_dir)

    # utils.init_distributed_mode(args)
    print(args)

    if args.distributed and args.dataparallel:
        raise RuntimeError("Distributed mode cannot be executed with Dataparallel mode .......!")

    cudnn.benchmark = True ## ??????????????????????????????????????????????????????????

    print(args.data_path.split('/')[-2].split('_')[-1])

    if args.distributed:
        mp.spawn(main, nprocs=len(args.device_ids), args=args)
    else:
        main(args)
