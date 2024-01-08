from torch import nn
import torch
import torch.nn.functional as F

class LassoLoss(nn.MSELoss):
    def __init__(
            self,
            named_parameters,
            l1: float,
            size_average=None,
            reduce=None,
            reduction="mean",
        ):
        super().__init__(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
        self.named_parameters = named_parameters
        self.l1 = l1
    
    def forward(self, input, target):
        loss = F.mse_loss(input, target, reduction=self.reduction)
        curr_norm = sum(
            torch.norm(p.data) for nm, p in self.named_parameters
            if nm.endswith(".weight")
        )
        loss = loss + self.l1 * curr_norm
        return loss