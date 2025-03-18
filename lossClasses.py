import torch as pt
import torch.nn as nn

class MyCrossEntropy(nn.Module):
    def __init__(self):
        super(MyCrossEntropy, self).__init__()
        self.eps = 1e-12

    def forward(self, preds:pt.Tensor, targets:pt.Tensor)->pt.Tensor:
        preds = pt.clamp(preds, min=self.eps, max=1-self.eps)
        loss = - targets * pt.log(preds) - (1 - targets) * pt.log(1 - preds)
        return loss.mean()


class MyRootMeanSquare(nn.Module):
    def __init__(self):
        super(MyRootMeanSquare, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = 1e-12

    def forward(self, preds:pt.Tensor, targets:pt.Tensor)->pt.Tensor:
        preds = pt.clamp(preds, min=self.eps, max=1-self.eps)
        mse = self.mse(preds, targets)
        rmse = pt.sqrt(mse)
        return rmse
