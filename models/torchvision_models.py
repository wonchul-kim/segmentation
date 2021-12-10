import torchvision
import torch

def create_model(args):
    if not args.weights:
        if args.pretrained:
            if args.model == 'deeplabv3_resnet101':
                model = torchvision.models.segmentation.__dict__[args.model](
                            pretrained=args.pretrained,
                            aux_loss=False#args.aux_loss,
                        )       
                model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, args.num_classes)
                model.aux_classifier[4] = torch.nn.Conv2d(256, args.num_classes, kernel_size=(1, 1), stride=(1, 1))
        else:
            model = torchvision.models.segmentation.__dict__[args.model](
                pretrained=args.pretrained,
                num_classes=args.num_classes,
                aux_loss=args.aux_loss,
            )
        print(">>> LODED TORCHVISION MODEL: ", args.model)
    
    return model