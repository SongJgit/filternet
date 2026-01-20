from __future__ import annotations

import torch

from filternet.registry import FILTER

from .kalman_filter import KalmanFilter

# torch.set_default_dtype(torch.float64)


@FILTER.register_module()
class ExtendedKalmanFilter(KalmanFilter):
    """observation function is decouple from filter by Params, so EKF is not
    changed compared to KF.

    Args:
        KalmanFilter (_type_): _description_
    """
    pass
