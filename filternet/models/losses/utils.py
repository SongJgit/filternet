import functools
from typing import Callable, Optional, Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from filternet.registry import MODELS
import numpy as np

# Modified from https://github.com/open-mmlab/mmdetection/tree/main/mmdet/models/losses


def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss: Tensor,
                       weight: Optional[Tensor] = None,
                       reduction: str = 'mean',
                       avg_factor: Optional[float] = None) -> Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Optional[Tensor], optional): Element-wise weights.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (Optional[float], optional): Average factor when
            computing the mean of losses. Defaults to None.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func: Callable) -> Callable:
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                reduction: str = 'mean',
                avg_factor: Optional[int] = None,
                **kwargs) -> Tensor:
        """
        Args:
            pred (Tensor): The prediction.
            target (Tensor): Target bboxes.
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            reduction (str, optional): Options are "none", "mean" and "sum".
                Defaults to 'mean'.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.

        Returns:
            Tensor: Loss tensor.
        """
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def build_loss(cfg: dict = None) -> Callable:
    """Build loss.
    Example:
        cfg = dict(
        loss_name='SmoothL1Loss', # or SmoothL1Loss
        params=dict(reduction='mean'),
    )
        loss = build_loss(cfg)
    Args:
        cfg (dict): The loss config, which should contain:
            - type (str): Module name.
            - loss_weight (float): Weight of loss.
            - args: Args for the loss module.
    Returns:
        Callable: A PyTorch loss.
    """

    if cfg is not None:
        try:
            loss_fn = MODELS.build(dict(type=cfg['loss_name'], **cfg['params']))
        except Exception as e:
            try:
                loss_fn = eval(f"nn.{cfg['loss_name']}")(**cfg['params'])
            except Exception:
                raise ValueError(f"Failed to build loss '{cfg['loss_name']}': {str(e)}. "
                                 f'Please check if the loss name is correct and all required '
                                 f"parameters are provided in cfg['params']")
    else:
        loss_fn = None
    return loss_fn


def common_collect4loss(preds: torch.Tensor,
                        tgts: torch.Tensor,
                        batch: Dict,
                        transforms: bool = False) -> Dict[str, torch.Tensor]:
    """_summary_

    Args:
        preds (torch.Tensor): [B, D, L]
        tgts (torch.Tensor): [B, D, L]
        batch (Dict): _description_
        transforms (bool, optional): _description_. Defaults to False.

    Returns:
        Dict[str, torch.Tensor]: _description_
    """
    pred_loss_mask = batch['pred_loss_mask']
    target_loss_mask = batch['target_loss_mask']
    valid_step_mask = batch['valid_step_mask']

    valid_masked_preds = preds[:, pred_loss_mask, :]
    valid_masked_targets = tgts[:, target_loss_mask, :]

    valid_step_mask = valid_step_mask[:, None, :].expand_as(valid_masked_preds)  # [B, T] -> [B, D, T]

    non_nan_mask = ~valid_masked_preds.isnan().any(dim=1, keepdim=True)
    non_inf_mask = ~valid_masked_preds.isinf().any(dim=1, keepdim=True)
    valid_mask = non_nan_mask & non_inf_mask & valid_step_mask.bool()
    valid_mask = valid_mask.float()

    avg_factor = valid_mask.sum()

    valid_preds = valid_masked_preds * valid_mask
    valid_targets = valid_masked_targets * valid_mask

    return dict(preds=valid_preds, tgts=valid_targets, avg_factor=avg_factor, valid_mask=valid_mask)
