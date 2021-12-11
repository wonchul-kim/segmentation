import torchvision
import torch

def torchvision_models(args):
    if args.pretrained:
        if args.model.split('_')[0] == 'deeplabv3':
            model = torchvision.models.segmentation.__dict__[args.model](pretrained=args.pretrained, aux_loss=args.aux_loss)
            if args.model.split('_')[1] == 'resnet101': 
                model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(model.backbone.layer4[2].conv3.out_channels, args.num_classes)
                model.aux_classifier[4] = torch.nn.Conv2d(256, args.num_classes, kernel_size=(1, 1), stride=(1, 1))
            elif args.model.split('_')[1] == 'resnet50': 
                model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(model.backbone.layer4[2].conv3.out_channels, args.num_classes)
                model.aux_classifier[4] = torch.nn.Conv2d(256, args.num_classes, kernel_size=(1, 1), stride=(1, 1))
            elif args.model.split('_')[1] == 'mobilenet': 
                model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(960, args.num_classes)
                model.aux_classifier[4] = torch.nn.Conv2d(10, args.num_classes, kernel_size=(1, 1), stride=(1, 1))
        elif args.model == 'lraspp_mobilenet_v3_large': 
            model = torchvision.models.segmentation.__dict__[args.model](pretrained=args.pretrained)
            model.classifier = torchvision.models.segmentation.lraspp.LRASPPHead(40, 960, args.num_classes, 128)
    else:
        model = torchvision.models.segmentation.__dict__[args.model](
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            aux_loss=args.aux_loss,
        )
    
    return model