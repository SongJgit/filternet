import math

import torch
from torch import Tensor
from filternet.utils.misc import check_nan_inf
from filternet.registry import MODELS
import torch.nn as nn
from typing import Optional
from .utils import weighted_loss


@weighted_loss
def compute_logpdf_Gaussian(pred: torch.Tensor,
                            target: torch.Tensor,
                            cov: torch.Tensor,
                            reduction: str = 'mean') -> torch.Tensor:
    """negative log-likelihood loss, modified from DANSE, "AssertionErrorGhosh,
    A., Honoré, A. & Chatterjee, S. DANSE: data-driven non-linear state
    estimation of model-free process in unsupervised learning setup. IEEE
    Trans. Signal Process. 1–14 (2024) doi:10.1109/TSP.2024.3383277.".

    Args:
        target (torch.Tensor): [bs, seq_len, n_dim], ground truth or observation.
        mean (torch.Tensor): mean of the distribution [bs, seq_len, n_dim].
        cov (torch.Tensor): covariance of the distribution [bs, seq_len, n_dim, n_dim].
        reduction (str): default is 'mean'. If reduction is 'none', then (bs).
    Returns:
        torch.Tensor: _description_
    """
    B, T, n_dim = target.shape
    # [bs]
    mask = (cov.abs().sum(dim=(-2, -1)) > 1e-6)
    inv_cov = torch.zeros_like(cov)
    logdet = torch.zeros(B, T, device=cov.device)

    # 添加小量保证数值稳定性
    stable_cov = cov[mask] + torch.eye(n_dim, device=cov.device) * 1e-6
    inv_cov[mask] = torch.linalg.inv(stable_cov)
    logdet[mask] = torch.logdet(stable_cov)

    residual = target - pred
    quad_term = torch.einsum('nti,nti->nt', residual, torch.einsum('ntij,ntj->nti', inv_cov, residual))

    logprob = 0.5 * n_dim * math.log(math.pi * 2) - 0.5 * logdet.sum(1) - 0.5 * quad_term.sum(1)
    return logprob


@MODELS.register_module()
class LogPDFGaussian(nn.Module):

    def __init__(self, reduction: str = 'mean', loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: torch.Tensor,
                target: Tensor,
                cov: torch.Tensor,
                reduction_override: Optional[str] = None) -> Tensor:
        """
        Args:
        pred (torch.Tensor): Mean of the distribution [B, D, T],
        target (torch.Tensor): ground truth or observation, [B, D, T].
        cov (torch.Tensor): covariance of the distribution [B, D, D, T].
        reduction (str): default is 'mean'. If reduction is 'none', then (bs).
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        pred = pred.permute(0, 2, 1)  # [B, D, T] -> [B, T, D]
        target = target.permute(0, 2, 1)  # [B, D, T] -> [B, T, D]
        cov = cov.permute(0, 3, 1, 2)  # [B, D, D, T] -> [B, T, D, D]
        loss = self.loss_weight * compute_logpdf_Gaussian(pred, target, cov=cov, reduction=reduction)
        return loss
