import datetime
import os
import os.path as osp
import time
import json
import torch
import torch.utils.data
from torch import nn
import torchvision

from coco_utils import get_coco
import presets
import utils
from tqdm import tqdm
from torchvision import transforms
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib 
from losses import auxiliar_loss, DiceLoss
from models.torchvision_models import torchvision_models
from models.unetpp import UNet, NestedUNet
import argparse
import numpy as np
from PIL import Image
import math
import cv2
import warnings
import glob
import copy
from transforms import Compose
import transforms as T

warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

IDS = [0, 64, 96, 128, 192, 248]
VALUES = [0., 1., 2., 3., 4., 5.]
t2l = { val : id_ for val, id_ in zip(VALUES, IDS) }

def get_transform(train, base_size, crop_size):
    return presets.SegmentationPresetDetect(base_size)

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

def save_as_images(img, tensor_pred, folder, image_name):

    if not os.path.exists(os.path.join(folder, 'tensors')):
        os.makedirs(os.path.join(folder, 'tensors'))
    filename = os.path.join(folder, image_name + '.png')

    fig = plt.figure(figsize=(30, 20), dpi=200)
    plt.subplot(121)
    plt.imshow(img)
    plt.xlabel("original")
    plt.subplot(122)
    plt.imshow(img)
    plt.imshow(tensor_pred, alpha=0.8)
    plt.xlabel("pred")
    plt.savefig(filename)
    plt.close()

    # tensor_pred.save(os.path.join(folder, 'tensors', image_name + '.png'))
    # np.save(os.path.join(folder, 'tensors', image_name + '.png'), tensor_pred)
    # print(type(tensor_pred))
    Image.fromarray(tensor_pred).save(os.path.join(folder, 'tensors', image_name + '.png'))


def evaluate(model, transform, device, num_classes, output_dir):
    offset1 = 0
    offset2 = 0
    model.eval()

    with torch.no_grad():
        cnt = 1
        img_files = glob.glob(osp.join(args.data_path, '*.jpg'))
        for img_file in tqdm(img_files):
            
            fname = osp.split(osp.splitext(img_file)[0])[-1]
            
            img = Image.open(img_file).convert("RGB")
            _img = copy.deepcopy(img)
            _img = _img.resize((1280, 1280))
            # img = cv2.resize(img, (1280, 1280))
            # _img = copy.deepcopy(img)
            # img = transforms.ToTensor()(img)
            img = transform(img)
            img = torch.unsqueeze(img, 0)
            img = img.to(device)

            output = model(img)
            output = output['out']

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
            preds_labels.apply_(lambda x: t2l[x])
            # preds_labels = transforms.Resize((1100, 1200), interpolation=Image.NEAREST)(preds_labels)
            # print(preds_labels.size(), np.unique(preds_labels.cpu()))
            
            cnt += 1
            preds_labels = np.array(transforms.ToPILImage()(preds_labels[0].byte()))

            # image, cx, cy, r = get_circle(im0, fname)
            # preds_labels = exceptions(preds_labels, cx, cy, r, offset1, offset2)
            save_as_images(_img, preds_labels, output_dir, fname)                



def main(args):
    device = torch.device(args.device)

    if args.model_name == 'deeplabv3_resnet101':
        model = torchvision_models(args.model_name, args.pretrained, args.loss, args.num_classes)

    # checkpoint = torch.load(args.weights, map_location='cpu')
    # model.load_state_dict(checkpoint['model'], strict=True)

    model = torch.load(args.weights)
    print(">>> Loaded the model: ", args.weights)
    model.to(device)
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)

    transform = T.Compose_([
        T.RandomResize_(args.base_imgsz, args.base_imgsz),
        T.ToTensor_(),
        T.Normalize_(mean=mean, std=std),
    ])

    evaluate(model, transform=transform, device=device, num_classes=args.num_classes, output_dir=args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data-path', 
        default='/home/wonchul/mnt/NAS/Data/01.Image/interojo/3rd_poc/21.11.04_C/인쇄확인필요/')
    parser.add_argument('--dataset-type', default='coco', help='dataset name')
    parser.add_argument('--model-name', default='deeplabv3_resnet101', help='model name')
    parser.add_argument("--pretrained", default=True)
    parser.add_argument('--num-classes', default=6, type=int, help='number of classes')
    parser.add_argument('--base-imgsz', default=1280, type=int, help='base image size')
    parser.add_argument('--crop-imgsz', default=1280, type=int, help='base image size')
    parser.add_argument('--output-dir', default='./outputs/detect')
    parser.add_argument('--loss', default='DiceLoss', help='loss type: None, BCEDiceLoss, aux_loss, DiceLoss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num-workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--weights', default='./outputs/train/2/weights/deeplabv3_resnet101_checkpoint_best.pt')

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
        
    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print(args)

    main(args)