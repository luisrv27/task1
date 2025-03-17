import torch as pt
import torch.nn as nn

class MyCrossEntropy(nn.Module):
    def __init__(self):
        super(MyCrossEntropy, self).__init__()

    def forward(self, preds, targets):
        preds = pt.clamp(preds, min=1e-5, max=1 - 1e-5)
        #print(preds)
        loss = - (targets * pt.log(preds) + (1 - targets) * pt.log(1 - preds))
        return pt.mean(loss)


class MyRootMeanSquare(nn.Module):
    def __init__(self):
        super(MyRootMeanSquare, self).__init__()

    def forward(self, preds, targets):
        preds = pt.clamp(preds, min=1e-5, max=1 - 1e-5)
        return pt.sqrt(pt.mean((preds - targets) ** 2))
