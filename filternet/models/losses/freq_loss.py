import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from filternet.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class FreqDomainLoss(nn.Module):

    def __init__(
        self,
        auxi_mode: str = 'fft',  # fft or rfft,
        auxi_type: str | None = 'complex',
        auxi_loss: str = 'MSE',
        module_first: bool = True,
        loss_weight: float = 1.0,
    ):
        """Modify from the paper :

        @inproceedings{wang2025fredf,
            title = {FreDF: Learning to Forecast in the Frequency Domain},
            author = {Wang, Hao and Pan, Licheng and Chen, Zhichao and Yang, Degui and Zhang, \
                Sen and Yang, Yifei and Liu, Xinggao and Li, Haoxuan and Tao, Dacheng},
            booktitle = {ICLR},
            year = {2025},
        }

        Args:
            auxi_mode (str, optional): _description_. Defaults to 'fft'.
            auxi_type (str | None, optional): _description_. Defaults to None.
            auxi_loss (str, optional): _description_. Defaults to 'MSE'.
            module_first (bool, optional): If True, first abs second mean, else first mean second abs. Defaults to True.
        """

        super().__init__()
        self.auxi_mode = auxi_mode
        self.auxi_type = auxi_type
        self.auxi_loss = auxi_loss
        self.module_first = module_first
        self.loss_weight = loss_weight

        assert self.auxi_mode in ['fft', 'rfft'], f"auxi_mode must be one of ['fft', 'rfft'], but got {self.auxi_mode}"
        if self.auxi_mode == 'rfft':
            assert self.auxi_type in [
                'complex', 'complex-phase', 'complex-mag-phase', 'phase', 'mag', 'mag-phase'
            ], f"auxi_type must be one of ['complex', 'complex-phase', 'complex-mag-phase', 'phase', 'mag', \
                'mag-phase' ], but got {self.auxi_type}"

        assert self.auxi_loss in ['MSE', 'MAE',
                                  'SmoothL1'], f"auxi_loss must be one of ['MSE', 'MAE'], but got {self.auxi_loss}"

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        avg_factor: Optional[int] = None,
    ) -> Tensor:
        """_summary_

        Args:
            pred (torch.Tensor): [B, D, L]
            target (torch.Tensor): [B, D, L]

        Returns:
            torch.Tensor: _description_
        """
        pred = pred.transpose(-1, -2)  # [B, D, L] -> [B, L, D]
        target = target.transpose(-1, -2)
        if self.auxi_mode == 'fft':
            loss = torch.fft.fft(pred, dim=1) - torch.fft.fft(target, dim=1)
        elif self.auxi_mode == 'rfft':
            if self.auxi_type == 'complex':
                loss = torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)
            elif self.auxi_type == 'complex-phase':
                loss = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).angle()
            elif self.auxi_type == 'complex-mag-phase':
                loss_mag = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs()
                loss_phase = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).angle()
                loss = torch.stack([loss_mag, loss_phase])
            elif self.auxi_type == 'phase':
                loss = torch.fft.rfft(pred, dim=1).angle() - torch.fft.rfft(target, dim=1).angle()
            elif self.auxi_type == 'mag':
                loss = torch.fft.rfft(pred, dim=1).abs() - torch.fft.rfft(target, dim=1).abs()
            elif self.auxi_type == 'mag-phase':
                loss_mag = torch.fft.rfft(pred, dim=1).abs() - torch.fft.rfft(target, dim=1).abs()
                loss_phase = torch.fft.rfft(pred, dim=1).angle() - torch.fft.rfft(target, dim=1).angle()
                loss = torch.stack([loss_mag, loss_phase])

        if avg_factor is None:
            avg_factor = pred.numel()
        else:
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            avg_factor = torch.finfo(torch.float32).eps + avg_factor

        if self.module_first:
            loss = loss.abs()

        if self.auxi_loss == 'MAE':
            # MAE, 最小化element-wise error的模长
            loss = loss.sum() / avg_factor  # check the dim of fft
        elif self.auxi_loss == 'MSE':
            # MSE, 最小化element-wise error的模长
            loss = (loss ** 2).sum() / avg_factor
        elif self.auxi_loss == 'SmoothL1':

            # 对实部和虚部分别应用SmoothL1
            if torch.is_complex(loss):
                real_loss = nn.SmoothL1Loss(reduction='sum')(loss.real, torch.zeros_like(loss.real)) / avg_factor
                imag_loss = nn.SmoothL1Loss(reduction='sum')(loss.imag, torch.zeros_like(loss.imag)) / avg_factor
                loss = (real_loss + imag_loss) / 2
            else:
                loss = nn.SmoothL1Loss(reduction='sum')(loss, torch.zeros_like(loss)) / avg_factor
        else:
            raise NotImplementedError

        if not self.module_first:
            loss = loss.abs()  # 后取模

        return self.loss_weight * loss
