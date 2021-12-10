from torchvision_models import torchvision_models
import unetpp 

TORCHVISION_MODELS = [
    'fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large',
    'lraspp_mobilenet_v3_large'
]

UNETPP = ['UNet', 'NestedUNet']

def CreateModel(args):
    if args.model in TORCHVISION_MODELS:
        model = torchvision_models(args)

        ### SET OPTIMIZER & SCHEDULER ------------------------------------------------------------------------------------------
        params_to_optimize = [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
        ]
        if args.aux_loss:
            params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
            params_to_optimize.append({"params": params, "lr": args.lr * 10})

        return model, params_to_optimize

    elif args.model in UNETPP:
        model = unetpp.__dict__[args.model](args.num_classes, args.input_channels, args.deep_supervision)

        return model, model.parameters()

