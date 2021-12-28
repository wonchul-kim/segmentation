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
from torchvision import transforms

from coco_utils import get_coco
import presets
import utils
from tqdm import tqdm

### moduel 
from models.models import CreateModel


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

### moduel 
from models.models import CreateModel


warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

IDS = [0, 64, 96, 128, 192, 248]
VALUES = [0., 1., 2., 3., 4., 5.]
t2l = { val : id_ for val, id_ in zip(VALUES, IDS) }

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

def export_libtorch(model, transform, device, args):
    model.eval()

    with torch.no_grad():
        img_files = glob.glob(osp.join('./images', '*.jpg'))
        for img_file in tqdm(img_files):
            fname = osp.split(osp.splitext(img_file)[0])[-1]
            
            img = Image.open(img_file).convert("RGB")
            _img = copy.deepcopy(img)
            _img = _img.resize((args.crop_imgsz, args.crop_imgsz))
            img = transform(img)
            img = torch.unsqueeze(img, 0)
            img = img.to(device)

            output = model(img)[0]

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
            plt.savefig(os.path.join(args.data_path, 'res', osp.split(osp.splitext(img_file)[0])[-1] + '_pred.jpg'))
            plt.close()

        traced_script_module = torch.jit.trace(model, (img))
        cfg = {"shape": img.shape, "classes": args.classes}
        print(cfg)
        extra_files = {'config': json.dumps(cfg)}  # torch._C.ExtraFilesMap()
        traced_script_module.save(osp.join(args.output_dir, 'libtorch_segmentation.pt'), _extra_files=extra_files)

def main(args):
    device = torch.device(args.device)

    model, params_to_optimize = CreateModel(args)
    print(">>> LODED TORCHVISION MODEL: ", args.model)

    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = wrapper(model)  

    print(">>> Loaded the model: ", args.weights)
    model.to(device)
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)

    transform = T.Compose_([
        T.RandomResize_(args.crop_imgsz, args.crop_imgsz),
        T.ToTensor_(),
        T.Normalize_(mean=mean, std=std),
    ])

    if not osp.exists(osp.join(args.data_path, 'res')):
        os.mkdir(osp.join(args.data_path, 'res'))
        args.data_path = osp.join(args.data_path, 'res')

    export_libtorch(model, transform=transform, device=device, args=args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')


    parser.add_argument('--model', default='deeplabv3_resnet101', help='model name')
    parser.add_argument("--pretrained", default=True)
    parser.add_argument('--data-path', default='./images', help='dataset path')
    parser.add_argument("--aux-loss", action="store_true", help="auxiliar loss")
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num-workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--output-dir', default='./outputs/libtorch')
    parser.add_argument("--input_dir", default='./outputs/train/seg11')
    

    args = parser.parse_args()

    args.weights = osp.join(args.input_dir, 'weights/last.pth')
    with open(osp.join(args.input_dir, 'cfg/config.json')) as cfg_file:
        json_data = json.load(cfg_file)
        args.classes = list(osp.split(osp.splitext(json_data['data_path'])[0])[-1].split("_"))
        args.base_imgsz = json_data['base_imgsz']
        args.crop_imgsz = json_data['crop_imgsz']
        args.num_classes = len(args.classes) + 1

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
