import torch
import torch.nn as nn
import numpy as np
from filternet.registry import MODELS


def divide_no_nan(a, b):
    """a/b where the resulted NaN or Inf are replaced by 0."""
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


@MODELS.register_module()
class MAPELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.tensor:
        """MAPE loss as defined in:

        https://en.wikipedia.org/wiki/Mean_absolute_percentage_error.
        """
        weights = divide_no_nan(torch.ones_like(target), target)
        return torch.mean(torch.abs((pred - target) * weights))


@MODELS.register_module()
class sMAPELoss(nn.Module):

    def __init__(self):
        super(sMAPELoss, self).__init__()

    def forward(
        self,
        forecast: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/
        (Makridakis 1993)"""
        return 200 * torch.mean(
            divide_no_nan(torch.abs(forecast - target),
                          torch.abs(forecast.data) + torch.abs(target.data)))
