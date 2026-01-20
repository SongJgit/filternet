import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import grad_norm
from typing import List
from filternet.param_schedulers import get_scheduler
from filternet.registry import MODELS, OPTIMIZER
from filternet.utils import MSE, MSEdB
from mmengine import Config


class BaseTrainer(pl.LightningModule):

    def __init__(self, cfg, save_dir: dict = None) -> None:
        super().__init__()
        self.model = MODELS.build(cfg.MODEL)
        self.cfg = cfg
        self.save_dir = save_dir
        self.vis_grad = self.cfg.TRAINER.vis_grad if hasattr(self.cfg.TRAINER, 'vis_grad') else True

        # select state to compute loss. dim must be equal.
        self.pred_metric_mask = cfg.METRIC.pred_metric_mask if cfg.METRIC.pred_metric_mask is not None else torch.ones(
            self.model.dim_state, dtype=torch.bool)
        self.target_metric_mask = cfg.METRIC.target_metric_mask if cfg.METRIC.target_metric_mask is not None else torch.ones(  # noqa: E501
            self.model.dim_state, dtype=torch.bool)

        self.automatic_optimization = False
        self._init_metric()

    def _init_metric(self):
        self.train_mse_dB = MSEdB()
        self.train_rmse = MSE(squared=False)

        self.test_mse_dB = MSEdB()
        self.test_rmse = MSE(squared=False)

        self.val_mse_dB = MSEdB()
        self.val_rmse = MSE(squared=False)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        if self.vis_grad:
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)

    def configure_optimizers(self):
        try:
            optimizer = eval(f'torch.optim.{self.cfg.OPTIMIZER.type}')(params=self.model.parameters(),
                                                                       **self.cfg.OPTIMIZER.params)
        except Exception as e:
            print(e)
            optimizer = OPTIMIZER.build(
                dict(type=self.cfg.OPTIMIZER['type'], params=self.model.parameters(), **self.cfg.OPTIMIZER['params']))

        if hasattr(self.cfg, 'SCHEDULER'):
            self.cfg.SCHEDULER.optimizer = optimizer
            scheduler = get_scheduler(self.cfg)
            self.cfg.SCHEDULER.pop('optimizer')
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        else:
            return {'optimizer': optimizer}

    def process_batch(self, batch, indices):
        _, _, seq_len = batch['targets'].shape
        # For slide window train, see https://github.com/SongJgit/KalmanNet4SensorFusion.
        # slide_win_size control the backward.
        slide_win_size = self.cfg.TRAINER.slide_win_size if self.cfg.TRAINER.slide_win_size else seq_len
        # if detach_step is None, will use slide_win_size as detach_step.
        detach_step = self.cfg.TRAINER.detach_step if self.cfg.TRAINER.detach_step else slide_win_size

        targets_per_win = batch['targets'][:, :, indices]  # [bs, num_state, win_size]

        inputs_per_win = batch['inputs'][:, :, indices]
        valid_step_mask = batch['valid_step_mask'][:, indices]

        in_batch = dict(initial_state=batch['initial_state'],
                        inputs=inputs_per_win,
                        targets=targets_per_win,
                        pred_loss_mask=self.cfg.LOSS.pred_loss_mask
                        if self.cfg.LOSS.pred_loss_mask else torch.ones(self.model.dim_state, dtype=torch.bool),
                        target_loss_mask=self.cfg.LOSS.target_loss_mask
                        if self.cfg.LOSS.target_loss_mask else torch.ones(self.model.dim_state, dtype=torch.bool),
                        valid_step_mask=valid_step_mask,
                        data_info=batch['data_info'],
                        detach_step=detach_step)

        return in_batch

    def process_slide_window(self, seq_len: int) -> List[torch.Tensor]:

        # For slide window train, see https://github.com/SongJgit/KalmanNet4SensorFusion.
        indices_win = torch.arange(0, seq_len)
        slide_win_size = self.cfg.TRAINER.slide_win_size if self.cfg.TRAINER.slide_win_size else seq_len
        indices_win = torch.split(indices_win, slide_win_size)
        return indices_win

    def on_save_checkpoint(self, checkpoint):
        checkpoint['cfg'] = self.cfg.to_dict()
