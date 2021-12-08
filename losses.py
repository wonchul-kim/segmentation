import torch.nn.functional as F
import torch.nn as nn 

#PyTorch
ALPHA = 0.5
BETA = 0.5
GAMMA = 1

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = inputs['out']
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky



class CELoss(nn.Module):
    def __init__(self, aux):
        super(CELoss, self).__init__()
        if aux == 'CE_AUX':
            self.aux = True

    def forward(self, inputs, target):
        losses = {}
        for name, x in inputs.items():
            losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

        if not self.aux:
            return losses["out"]
        else:
            return losses["out"] + 0.5 * losses["aux"]

# def CELoss(inputs, target):
#     losses = {}
#     for name, x in inputs.items():
#         losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

#     if len(losses) == 1:
#         return losses["out"]

#     return losses["out"] + 0.5 * losses["aux"]