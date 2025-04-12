import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import grad_norm

from filternet.param_schedulers import get_scheduler
from filternet.registry import MODELS, OPTIMIZER
from filternet.utils import MSE, MSEdB


class BaseTrainer(pl.LightningModule):

    def __init__(self, cfg, save_dir: dict) -> None:
        super().__init__()
        self.model = MODELS.build(cfg.MODEL)
        self.cfg = cfg
        self.save_dir = save_dir
        self.vis_grad = self.cfg.TRAINER.vis_grad if hasattr(
            self.cfg.TRAINER, 'vis_grad') else True

        # select state to compute loss. dim must be equal.
        self.pred_metric_mask = cfg.METRIC.pred_metric_mask if cfg.METRIC.pred_metric_mask is not None else torch.ones(
            self.model.dim_state, dtype=torch.bool)
        self.target_metric_mask = cfg.METRIC.target_metric_mask if cfg.METRIC.target_metric_mask is not None else torch.ones(  # noqa: E501
            self.model.dim_state, dtype=torch.bool)

        self.train_mse_dB = MSEdB()
        self.train_rmse = MSE(squared=False)

        self.test_mse_dB = MSEdB()
        self.test_rmse = MSE(squared=False)

        self.val_mse_dB = MSEdB()
        self.val_rmse = MSE(squared=False)

        self.automatic_optimization = False

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        if self.vis_grad:
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)

    def configure_optimizers(self):
        try:
            optimizer = eval(f'torch.optim.{self.cfg.OPTIMIZER.type}')(
                params=self.model.parameters(), **self.cfg.OPTIMIZER.params)
        except Exception as e:
            print(e)
            optimizer = OPTIMIZER.build(
                dict(type=self.cfg.OPTIMIZER['type'],
                     params=self.model.parameters(),
                     **self.cfg.OPTIMIZER['params']))

        if hasattr(self.cfg, 'SCHEDULER'):
            self.cfg.SCHEDULER.optimizer = optimizer
            scheduler = get_scheduler(self.cfg)
            self.cfg.SCHEDULER.pop('optimizer')

        if scheduler is not None:
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        else:
            return {'optimizer': optimizer}
