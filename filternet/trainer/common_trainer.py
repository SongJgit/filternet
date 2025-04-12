from typing import Any, Dict

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from filternet.registry import MODELS

from .base_trainer import BaseTrainer


@MODELS.register_module()
class CommonTrainer(BaseTrainer):

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        self.model.init_beliefs(batch['initial_state'])
        opt = self.optimizers()
        self.batch_size, _, seq_len = batch['targets'].shape

        # For slide window train, see https://github.com/SongJgit/KalmanNet4SensorFusion.
        indices_win = torch.arange(0, seq_len)
        slide_win_size = self.cfg.TRAINER.slide_win_size if self.cfg.TRAINER.slide_win_size is not None else seq_len
        detach_step = self.cfg.TRAINER.detach_step if self.cfg.TRAINER.detach_step is not None else slide_win_size
        indices_win = [
            indices_win[:slide_win_size],
            *torch.split(indices_win[slide_win_size:], slide_win_size)
        ]
        indices_win = indices_win[:-1] if len(
            indices_win[-1]) == 0 else indices_win

        # prepare to predict
        preds = torch.zeros(self.batch_size, self.model.dim_state,
                            seq_len).to(self.device)
        loss_item = 0
        loss4log: Dict[Any] = dict()  # type: ignore
        for i, indices in enumerate(indices_win):

            targets_per_win = batch[
                'targets'][:, :, indices]  # [bs, num_state, win_size]

            inputs_per_win = batch['inputs'][:, :, indices]
            valid_step_mask = batch['valid_step_mask'][:, indices]

            in_batch = dict(initial_state=batch['initial_state'],
                            inputs=inputs_per_win,
                            targets=targets_per_win,
                            pred_loss_mask=self.cfg.LOSS.pred_loss_mask,
                            target_loss_mask=self.cfg.LOSS.target_loss_mask,
                            valid_step_mask=valid_step_mask,
                            detach_step=detach_step)
            train_losses, out = self.model.loss_and_filtering(in_batch)
            train_loss = sum(
                [v for k, v in train_losses.items() if k.startswith('loss_')])

            loss_item += train_loss.item()
            opt.zero_grad()
            self.manual_backward(train_loss, retain_graph=True)
            self.clip_gradients(
                opt,
                gradient_clip_val=self.cfg.TRAINER.gradient_clip_val,
                gradient_clip_algorithm=self.cfg.TRAINER.
                gradient_clip_algorithm)
            opt.step()
            self.model._detach()

            for idx, (loss_name,
                      loss_value) in enumerate(train_losses.items()):
                if loss_name.startswith('loss_'):
                    if loss_name in loss4log.keys():
                        loss4log[loss_name] += loss_value.item()
                    else:
                        loss4log[loss_name] = loss_value.item()

            preds[:, :, indices] = out.detach()

        self.log('train_loss/sum_per_win',
                 loss_item / len(indices_win),
                 batch_size=self.batch_size)
        self.log('train_loss/sum_per_timestep',
                 loss_item / seq_len,
                 batch_size=self.batch_size)
        for loss_name, loss_value in loss4log.items():
            self.log(f"train_loss/{loss_name.strip('loss_')}_per_win",
                     loss_value / len(indices_win),
                     batch_size=self.batch_size)
            self.log(f"train_loss/{loss_name.strip('loss_')}_per_timestep",
                     loss_value / seq_len,
                     batch_size=self.batch_size)

        # For Metric

        all_pred = preds[:, self.pred_metric_mask, :]
        all_target = batch['targets'][:, self.target_metric_mask, :]
        if self.cfg.METRIC.inverse_transforms:
            # INFO:
            all_pred = self.trainer.datamodule.train.inverse_data(
                all_pred, batch)
            all_target = self.trainer.datamodule.train.inverse_data(
                all_target, batch)
        all_mask = batch['valid_step_mask'][:, None, :]

        masked_pred = torch.masked_select(
            all_pred, all_mask)  # must match, else error broadcast.
        masked_target = torch.masked_select(all_target, all_mask)

        self.train_mse_dB.update(masked_pred, masked_target)
        # self.train_axis_mse_dB.update(pred, target)
        self.train_rmse.update(masked_pred, masked_target)

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

        self.log('train/MSE[dB]',
                 self.train_mse_dB,
                 prog_bar=False,
                 batch_size=self.batch_size)
        self.log('train/RMSE',
                 self.train_rmse,
                 prog_bar=True,
                 batch_size=self.batch_size)

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        self.model.init_beliefs(batch['initial_state'])
        self.batch_size, dim_state, seq_len = batch['targets'].shape
        preds = torch.zeros(batch['initial_state'].shape[0],
                            self.model.dim_state, seq_len).to(self.device)

        in_batch = dict(
            initial_state=batch['initial_state'],
            inputs=batch['inputs'],
            targets=batch['targets'],
            pred_loss_mask=self.cfg.LOSS.pred_loss_mask,
            target_loss_mask=self.cfg.LOSS.target_loss_mask,
            valid_step_mask=batch['valid_step_mask'],
        )
        val_losses, preds = self.model.loss_and_filtering(in_batch)
        val_loss = sum(
            [v for k, v in val_losses.items() if k.startswith('loss_')])
        self.log('val_loss/loss', val_loss, batch_size=self.batch_size)

        for loss_name, loss_value in val_losses.items():
            if loss_name.startswith('loss_'):
                self.log(f"val_loss/{loss_name.strip('loss_')}",
                         loss_value,
                         batch_size=self.batch_size)

        # use mask to select axis and unpadded.
        all_pred = preds[:, self.pred_metric_mask, :]
        all_target = batch['targets'][:, self.
                                      target_metric_mask, :]  # type: ignore
        if self.cfg.METRIC.inverse_transforms:
            # INFO:
            all_pred = self.trainer.datamodule.val.inverse_data(
                all_pred, batch)
            all_target = self.trainer.datamodule.val.inverse_data(
                all_target, batch)
        masked_preds = torch.masked_select(
            all_pred, batch['valid_step_mask']
            [:, None, :])  # must match, else error broadcast.
        masked_targets = torch.masked_select(
            all_target, batch['valid_step_mask'][:, None, :])
        # For metric
        # Inverse data

        self.val_mse_dB.update(masked_preds, masked_targets)
        self.val_rmse.update(masked_preds, masked_targets)

        return val_loss

    def on_validation_epoch_end(self) -> None:
        self.log('val/MSE[dB]',
                 self.val_mse_dB,
                 prog_bar=True,
                 batch_size=self.batch_size)
        self.log('val/RMSE',
                 self.val_rmse,
                 prog_bar=True,
                 batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        self.model.init_beliefs(batch['initial_state'])
        self.batch_size, dim_state, seq_len = batch['targets'].shape
        # preds = torch.zeros(batch_size, dim_state, seq_len).to(self.device)

        in_batch = dict(
            initial_state=batch['initial_state'],
            inputs=batch['inputs'],
            targets=batch['targets'],
            pred_loss_mask=self.cfg.LOSS.pred_loss_mask,
            target_loss_mask=self.cfg.LOSS.target_loss_mask,
            valid_step_mask=batch['valid_step_mask'],
        )
        test_losses, preds = self.model.loss_and_filtering(in_batch)
        test_loss = sum(
            [v for k, v in test_losses.items() if k.startswith('loss_')])

        self.log('test_loss/loss', test_loss, batch_size=self.batch_size)
        for loss_name, loss_value in test_losses.items():
            if loss_name.startswith('loss_'):
                self.log(f"test_loss/{loss_name.strip('loss_')}",
                         loss_value,
                         batch_size=self.batch_size)

        all_pred = preds[:, self.pred_metric_mask, :]
        all_target = batch['targets'][:, self.
                                      target_metric_mask, :]  # type: ignore

        if self.cfg.METRIC.inverse_transforms:
            # INFO:
            all_pred = self.trainer.datamodule.train.inverse_data(
                all_pred, batch)
            all_target = self.trainer.datamodule.train.inverse_data(
                all_target, batch)
        masked_preds = torch.masked_select(
            all_pred, batch['valid_step_mask']
            [:, None, :])  # must match, else error broadcast.
        masked_targets = torch.masked_select(
            all_target, batch['valid_step_mask'][:, None, :])

        self.test_mse_dB.update(masked_preds, masked_targets)
        self.test_rmse.update(masked_preds, masked_targets)

        return test_loss

    def on_test_epoch_end(self) -> None:

        self.log('test/MSE[dB]', self.test_mse_dB, batch_size=self.batch_size)
        self.log('test/RMSE', self.test_rmse, batch_size=self.batch_size)

    def predict_step(self,
                     batch: Any,
                     batch_idx: int,
                     dataloader_idx: int = 0) -> Any:
        batch_size, _, len = batch['targets'].shape
        preds = torch.zeros(batch_size, self.model.dim_state,
                            len).to(self.device)
        self.model.init_beliefs(batch['initial_state'])
        valid_step_mask = batch['valid_step_mask']
        preds = self.model.filtering(batch)
        # process
        targets = []
        results = []
        inputs = []
        for i in range(batch_size):
            targets.append(
                torch.masked_select(batch['targets'][[i], ...],
                                    valid_step_mask[i][None, ...]).reshape(
                                        self.model.dim_state, -1).cpu())
            results.append(
                torch.masked_select(preds[[i], ...],
                                    valid_step_mask[i][None, ...]).reshape(
                                        self.model.dim_state, -1).cpu())
            inputs.append(
                torch.masked_select(batch['inputs'][[i], ...],
                                    valid_step_mask[i][None, ...]).reshape(
                                        batch['inputs'].shape[1], -1).cpu())

        return dict(preds=results, targets=targets, inputs=inputs)
