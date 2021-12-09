import torch.nn.functional as F
import torch.nn as nn 
import torch 
import numpy as np

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

# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()
        
#     def forward(self, preds, targets, smooth = 1e-5):
#         preds = preds['out']

#         print(preds.size(), targets.size())
#         print(torch.where(targets == 255, targets, 0))
#         targets = F.one_hot(targets).float()
#         print(targets.size())
#         targets = targets.transpose(1, 3)
#         print(targets.size())
#         targets = targets.transpose(2, 3)
#         print(targets.size())
        
#         bce = F.binary_cross_entropy_with_logits(preds, targets, reduction='sum')
        
#         preds = torch.sigmoid(preds)
#         intersection = (preds * targets).sum(dim=(2,3))
#         union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        
#         # dice coefficient
#         dice = 2.0 * (intersection + smooth) / (union + smooth)
        
#         # dice loss
#         dice_loss = 1.0 - dice
        
#         # total loss
#         loss = bce + dice_loss
    
#         return loss.sum(), dice.sum()

class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

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
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.num_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.num_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]

        return loss / self.num_classes