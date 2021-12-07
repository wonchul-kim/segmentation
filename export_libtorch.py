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
import argparse
import numpy as np
from PIL import Image
import math
import cv2
import warnings
import glob
import copy
from transforms_ import Compose
import transforms_ as T

warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

IDS = [0, 64, 96, 128, 192, 248]
VALUES = [0., 1., 2., 3., 4., 5.]
t2l = { val : id_ for val, id_ in zip(VALUES, IDS) }


def export(model, transform, device, num_classes, output_dir):
    model.eval()

    with torch.no_grad():
        img_file = './images/bubbles.jpg'
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

        output = model(img)[0]
        # print(output)
        # print(len(output))
        # output = output['out']

        preds = torch.nn.functional.softmax(output, dim=1)
        preds_labels = torch.argmax(preds, dim=1)
        preds_labels = preds_labels.float()
        # print("* pred size: ", preds_labels.size())
        
        # for x in range(preds_labels.size(1)):
        #     for y in range(preds_labels.size(2)):
        #         if preds_labels[0][x][y].cpu().detach().item() != 0.0:

        preds_labels = preds_labels.to('cpu')
        _, x, y = preds_labels.size()
        preds_labels.apply_(lambda x: t2l[x])
        # preds_labels = transforms.Resize((1100, 1200), interpolation=Image.NEAREST)(preds_labels)
        # print(preds_labels.size(), np.unique(preds_labels.cpu()))
        
        preds_labels = np.array(transforms.ToPILImage()(preds_labels[0].byte()))

        fig = plt.figure(dpi=150)
        plt.imshow(_img)
        plt.imshow(preds_labels, alpha=0.8)
        plt.xlabel("pred")
        plt.savefig(os.path.join('./images/pred_bubbles.png'))
        plt.close()

        traced_script_module = torch.jit.trace(model, (img))
        traced_script_module.save(osp.join(output_dir, 'libtorch_segmentation.pt'))

def main(args):
    class wrapper(torch.nn.Module):
        def __init__(self, model):
            super(wrapper, self).__init__()
            self.model = model

        def forward(self, input):
            results = []
            output = self.model(input)

            for k, v in output.items():
                results.append(v)
                return results

    device = torch.device(args.device)

    if args.model == 'deeplabv3_resnet101':
        model = torchvision.models.segmentation.__dict__[args.model](
                    pretrained=True,
                    aux_loss=args.aux_loss,
                )        
        model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, args.num_classes)
        model.aux_classifier[4] = torch.nn.Conv2d(256, args.num_classes, kernel_size=(1, 1), stride=(1, 1))

    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = wrapper(model)  

    print(">>> Loaded the model: ", args.weights)
    model.to(device)
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)

    transform = T.Compose_([
        T.RandomResize_(args.base_imgsz, args.base_imgsz),
        T.ToTensor_(),
        T.Normalize_(mean=mean, std=std),
    ])

    export(model, transform=transform, device=device, num_classes=6, output_dir=args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')


    parser.add_argument('--model', default='deeplabv3_resnet101', help='model name')
    parser.add_argument("--pretrained", default=True)
    parser.add_argument('--num-classes', default=6, type=int, help='number of classes')
    parser.add_argument('--base-imgsz', default=1280, type=int, help='base image size')
    parser.add_argument('--crop-imgsz', default=1280, type=int, help='base image size')
    parser.add_argument('--output-dir', default='./outputs/libtorch')
    parser.add_argument("--aux-loss", action="store_true", help="auxiliar loss")
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num-workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--weights', default='./outputs/train/1/weights/last.pth')

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
