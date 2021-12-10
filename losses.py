import torch.nn.functional as F
import torch.nn as nn 
import torch 
import numpy as np
try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

class CELoss(nn.Module):
    def __init__(self, aux_loss):
        super(CELoss, self).__init__()
        self.aux_loss = aux_loss

    def forward(self, preds, targets):
        losses = {}
        print(">>> {}, {}".format(preds['out'].size(), targets.size()))

        for name, x in preds.items():
            losses[name] = nn.functional.cross_entropy(x, targets, ignore_index=255)

        if not self.aux_loss:
            return losses["out"]
        else:
            return losses["out"] + 0.5 * losses["aux"]

class DiceLoss(nn.Module):
    def __init__(self, num_classes, bce_loss):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.bce_loss = bce_loss

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)

        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss

        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        inputs = inputs['out']
        target = self._one_hot_encoder(target)
        if self.bce_loss:
            bce = F.binary_cross_entropy_with_logits(inputs, target)

        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if weight is None:
            weight = [1] * self.num_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.num_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        
        if self.bce_loss:
            loss += bce
        return loss / self.num_classes




# class BCEDiceLoss(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.num_classes = num_classes

#     def _one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.num_classes):
#             temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob.unsqueeze(1))
#         output_tensor = torch.cat(tensor_list, dim=1)

#         return output_tensor.float()

#     def forward(self, input, target):
#         input = input['out']
#         target = self._one_hot_encoder(target)

#         bce = F.binary_cross_entropy_with_logits(input, target)
#         # bce = F.cross_entropy(input, target, ignore_index=255)
#         smooth = 1e-5
#         input = torch.sigmoid(input)
#         num = target.size(0)
#         input = input.view(num, -1)
#         target = target.view(num, -1)
#         intersection = (input * target)
#         dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
#         dice = 1 - dice.sum() / num
#         return 0.5 * bce + dice


# class LovaszHingeLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, input, target):
#         input = input.squeeze(1)
#         target = target.squeeze(1)
#         loss = lovasz_hinge(input, target, per_image=True)

#         return loss