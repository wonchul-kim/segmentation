import datetime
import os
import os.path as osp
import sys 
sys.path.append(osp.join(osp.dirname(__file__), 'src'))
sys.path.append(osp.join(osp.dirname(__file__), 'models'))

import time
import json
import torch
import torch.utils.data
from torch import nn
import torchvision

from coco_utils import get_coco
import presets
import utils

from torchvision import transforms
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib 

### moduel 
from models.models import CreateModel

from losses  import CELoss, DiceLoss

import argparse
import numpy as np
from PIL import Image
import math
import cv2
import warnings
warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_dataset(dir_path, dataset_type, mode, transform, num_classes):
    paths = {
        "coco": (dir_path, get_coco, num_classes),
    }

    ds_path, ds_fn, num_classes = paths[dataset_type]
    ds = ds_fn(ds_path, mode=mode, transforms=transform, num_classes=num_classes)
    return ds, num_classes

def get_transform(train, base_size, crop_size):
    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)

def get_circle(img, fname):
    np_img = np.array(transforms.ToPILImage()(img[0]))
    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blr = cv2.GaussianBlur(gray, (0, 0), 1)
    # circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1, 50,
    #                             param1=150, param2=40, minRadius=400,maxRadius=500)
    # circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1, 50,
    #                             param1=150, param2=40, minRadius=500,maxRadius=600)

    # if len(circles) != 1:
    #     print("ERROR for the number of circles: ", fname)
    # else:
    #     c = np.array(circles[0][0], np.int32)
    #     img_ = cv2.circle(np_img, (c[0], c[1]), c[2], (255, 0, 0), 4)
    #     print(img_.shape)
    #     plt.scatter(c[0], c[1], c='r')
    #     plt.imshow(img_)
    #     plt.show()

    return cv_img, c[0], c[1], c[2]

def exceptions(tensor_pred, cx, cy, r, offset1, offset2):
    # tensor_pred = np.array(transforms.ToPILImage()(tensor_pred[0].byte()))
    idxes = np.where(tensor_pred == 64)
    # print(tensor_pred.shape)
    for x, y in zip(idxes[0], idxes[1]):
        if math.sqrt((cx - x)**2 + (cy - y)**2) > r - offset1//2 and \
            math.sqrt((cx - x)**2 + (cy - y)**2) < r + offset1:
            tensor_pred[idxes[0], idxes[1]] = 0
        
        if math.sqrt((cx - x)**2 + (cy - y)**2) > r + offset2:
            tensor_pred[idxes[0], idxes[1]] = 0


    # tensor_pred = cv2.circle(tensor_pred, (cx, cy), r - offset1//2, (255, 0, 0), 1)
    # tensor_pred = cv2.circle(tensor_pred, (cx, cy), r + offset1, (255, 0, 0), 1)
    # plt.imshow(tensor_pred)
    # plt.show()

    return tensor_pred

def save_as_images(img, img1, target, tensor_pred, folder, image_name):
    if not os.path.exists(os.path.join(folder, 'tensors')):
        os.makedirs(os.path.join(folder, 'tensors'))
    # img = transforms.ToPILImage()(img[0].byte())
    # target = transforms.Resize((1200, 1600))(target)
    # print(np.unique(tensor_pred.cpu().numpy()))
    # np.save(os.path.join(folder, 'tensors', image_name + '.npy'), tensor_pred.cpu().numpy())
    filename = os.path.join(folder, image_name + '.png')

    # fig = plt.figure(figsize=(20, 20), dpi=150)
    # plt.subplot(131)
    # plt.imshow(img)
    # plt.xlabel("original")
    # plt.subplot(132)
    # plt.imshow(target)
    # plt.xlabel("target")
    # plt.subplot(133)
    # plt.imshow(tensor_pred)
    # plt.xlabel("pred")
    # plt.savefig(filename)
    # plt.close()

    fig = plt.figure(figsize=(30, 20), dpi=200)
    plt.subplot(121)
    plt.imshow(img)
    plt.xlabel("original")
    plt.subplot(122)
    plt.imshow(img1)
    plt.imshow(tensor_pred, alpha=0.8)
    plt.xlabel("pred")
    plt.savefig(filename)
    plt.close()

    # tensor_pred.save(os.path.join(folder, 'tensors', image_name + '.png'))
    # np.save(os.path.join(folder, 'tensors', image_name + '.png'), tensor_pred)
    # print(type(tensor_pred))
    Image.fromarray(tensor_pred).save(os.path.join(folder, 'tensors', image_name + '.png'))


def evaluate(args, model, data_loader, device, num_classes, output_dir):
    offset1 = 0
    offset2 = 0
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        cnt = 1
        for image, target, fname in metric_logger.log_every(data_loader, 1, header):

            im0 = cv2.imread(osp.join(args.data_path, 'val', 'Visualization', fname[0] + '.jpg'))
            im0 = cv2.resize(im0, (1280, 1280))

            im1 = cv2.imread(osp.join(args.data_path, 'val', 'JPEGImages', fname[0] + '.jpg'))
            im1 = cv2.resize(im1, (1280, 1280))

            image, target = image.to(device), target.to(device)
            # print("* input size: ", image.size(), target.size())
            output = model(image)
            output = output['out']
            # print("* output size: ", output.size())

            preds = torch.nn.functional.softmax(output, dim=1)
            preds_labels = torch.argmax(preds, dim=1)
            preds_labels = preds_labels.float()
            # print("* pred size: ", preds_labels.size())
            
            # for x in range(preds_labels.size(1)):
            #     for y in range(preds_labels.size(2)):
            #         print("\r {}, {}".format(x, y), end='')
            #         if preds_labels[0][x][y].cpu().detach().item() != 0.0:
            #             print('---', preds_labels[0][x][y].cpu().detach().item())

            preds_labels = preds_labels.to('cpu')
            _, x, y = preds_labels.size()
            preds_labels.apply_(lambda x: args.t2l[x])
            # preds_labels = transforms.Resize((1200, 1600), interpolation=Image.NEAREST)(preds_labels)
            # print(preds_labels.size())
            # preds_labels = transforms.Resize((1100, 1200), interpolation=Image.NEAREST)(preds_labels)
            # print(preds_labels.size(), np.unique(preds_labels.cpu()))
            
            # image = transforms.Resize((1200, 1600))(image)
            confmat.update(target.flatten(), output.argmax(1).flatten())

            cnt += 1
            preds_labels = np.array(transforms.ToPILImage()(preds_labels[0].byte()))
            target = transforms.ToPILImage()(target[0].byte())

            # image, cx, cy, r = get_circle(im0, fname)
            # preds_labels = exceptions(preds_labels, cx, cy, r, offset1, offset2)
            save_as_images(im0, im1, target, preds_labels, output_dir, fname[0])                

            # print(target.size(), image.size())
            # print("-----------------------------------------------") 
            # print(target.flatten().size(), target.size())
            # print(target.flatten())
            # print(output.argmax(1))

            # if cnt > 3:
            #     break

        confmat.reduce_from_all_processes()

    return confmat


def main(args):
    # utils.init_distributed_mode(args)
    device = torch.device(args.device)

    dataset_test, _ = get_dataset(args.data_path, args.dataset_type, "val", 
                                  get_transform(False, args.base_imgsz, args.crop_imgsz), args.num_classes)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

    model, params_to_optimize = CreateModel(args)

    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    print(">>>> RELOAD ALL !!!!!!!!!!!!!!!!!!!!!!!!!!")    # model.load_state_dict(checkpoint['model'], strict=True)

    model.to(device)

    confmat = evaluate(args, model, data_loader_test, device=device, num_classes=args.num_classes, output_dir=args.output_dir)

    print(confmat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data-path', default='/home/wonchul/HDD/datasets/projects/interojo/S_factory/coco_datasets_good/DUST_BUBBLE_DAMAGE_EDGE_RING_LINE_OVERLAP', help='dataset path')
    parser.add_argument('--dataset-type', default='coco', help='dataset name')
    parser.add_argument('--model', default='deeplabv3_resnet101', help='model name')
    parser.add_argument("--aux-loss", action="store_true", help="auxiliar loss")
    parser.add_argument("--pretrained", default=True)
    parser.add_argument('--num-classes', default=8, type=int, help='number of classes')
    parser.add_argument('--base-imgsz', default=1280, type=int, help='base image size')
    parser.add_argument('--crop-imgsz', default=1280, type=int, help='base image size')
    parser.add_argument('--output-dir', default='./outputs/val')
    parser.add_argument('--loss', default='DiceLoss', help='loss type: None, BCEDiceLoss, aux_loss, DiceLoss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num-workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--weights', default='./outputs/train/seg2/weights/best.pth')

    args = parser.parse_args()

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            args.output_dir = os.path.join(args.output_dir, '1')
            os.makedirs(args.output_dir)
        else:
            folders = sorted(list(map(int, os.listdir(args.output_dir))))
            if len(folders) != 0:
                args.output_dir = os.path.join(args.output_dir, str(folders[-1] + 1))
            else:
                args.output_dir = os.path.join(args.output_dir, str(1))
            os.makedirs(args.output_dir)
            
        output_path = args.output_dir
        args.date = str(datetime.datetime.now())
        
    IDS, VALUES = [], []
    diff = int(255//args.num_classes)
    for idx in range(args.num_classes):
        IDS.append(int(diff*idx))
        VALUES.append(idx)
    args.t2l = { val : id_ for val, id_ in zip(VALUES, IDS) }

    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print(args)

    main(args)