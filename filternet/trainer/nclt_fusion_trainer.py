import torch
from filternet.registry import MODELS
from typing import Any
from .common_trainer import CommonTrainer


@MODELS.register_module()
class NCLTFusionTrainer(CommonTrainer):

    def process_batch(self, batch, indices):
        _, _, seq_len = batch['targets'].shape
        # For slide window train, see https://github.com/SongJgit/KalmanNet4SensorFusion.
        # slide_win_size control the backward.
        slide_win_size = self.cfg.TRAINER.slide_win_size if self.cfg.TRAINER.slide_win_size else seq_len
        # if detach_step is None, will use slide_win_size as detach_step.
        detach_step = self.cfg.TRAINER.detach_step if self.cfg.TRAINER.detach_step else slide_win_size

        targets_per_win = batch['targets'][..., indices][:, self.target_metric_mask, :]  # [bs, num_state, win_size]
        inputs_per_win = batch[self.model.params.sensor_based][:, :, indices]
        correction = batch['filtered_gps'][:, :, indices]
        valid_step_mask = batch['valid_step_mask'][:, indices]

        in_batch = dict(initial_state=batch['initial_state'],
                        inputs=inputs_per_win,
                        targets=targets_per_win,
                        correction=correction,
                        pred_loss_mask=self.cfg.LOSS.pred_loss_mask
                        if self.cfg.LOSS.pred_loss_mask else torch.ones(self.model.dim_state, dtype=torch.bool),
                        target_loss_mask=self.cfg.LOSS.target_loss_mask
                        if self.cfg.LOSS.target_loss_mask else torch.ones(self.model.dim_state, dtype=torch.bool),
                        valid_step_mask=valid_step_mask,
                        detach_step=detach_step)

        return in_batch

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        self.model.init_beliefs(batch['initial_state'])

        in_batch = dict(inputs=batch[self.model.params.sensor_based], correction=batch['filtered_gps'])

        preds = self.model.filtering(in_batch)

        preds = preds[:, self.pred_metric_mask, :]

        targets: torch.Tensor = batch['targets']

        res = dict(preds=preds, targets=targets)
        res.update(batch)
        return res
