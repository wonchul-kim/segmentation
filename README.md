# For semantic segmentation

### ToDo

- [ ] distributed mode to use gpu fully and fairly
- [ ] add loss functions
- [ ] add other architectures


### Available models

#### torchvision models

* deeplabv3_resnet50

* deeplabv3_resnet101

* deeplabv3_mobilenet_v3_large
    > The loss func. should be `DiceLoss`. 

    > If the loss func. is `CE`, there is option with `aux_loss` argument.

* lraspp_mobilenet_v3_large
    > The loss func. should be `DiceLoss`. 

    > If the loss func. is `CE`, there is no option with `aux_loss` argument.



#### Nested Unet++

* NestedUnet

* UNet

    > `BceDiceLoss` is used in the paper.

    > Consider `deep-supervision` argument for NestedUnet