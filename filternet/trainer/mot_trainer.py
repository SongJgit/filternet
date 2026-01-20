from typing import Any, Dict

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics.detection import IntersectionOverUnion as IOU
from torchmetrics.detection import MeanAveragePrecision as mAP

from filternet.registry import MODELS
from .common_trainer import CommonTrainer
from filternet.utils.mot_utils import collect_mot_results_for_metric
from copy import deepcopy


@MODELS.register_module()
class MOTTrainer(CommonTrainer):

    def process_batch(self, batch, indices):
        bs, _, seq_len = batch['targets'].shape
        # For slide window train, see https://github.com/SongJgit/KalmanNet4SensorFusion.
        # slide_win_size control the backward.
        slide_win_size = self.cfg.TRAINER.slide_win_size if self.cfg.TRAINER.slide_win_size else seq_len
        targets_per_win = batch['targets'][:, :, indices]  # [bs, num_state, win_size]

        inputs_per_win = batch['inputs'][:, :, indices]

        valid_step_mask = batch['valid_step_mask'][:, indices]
        detach_step = self.cfg.TRAINER.detach_step if self.cfg.TRAINER.detach_step else slide_win_size
        data_info = deepcopy(batch['data_info'])
        for i in range(bs):
            data_info[i]['frame_ids'] = batch['data_info'][i]['frame_ids'][indices]
        in_batch = dict(initial_state=batch['initial_state'],
                        inputs=inputs_per_win,
                        targets=targets_per_win,
                        pred_loss_mask=self.cfg.LOSS.pred_loss_mask,
                        target_loss_mask=self.cfg.LOSS.target_loss_mask,
                        valid_step_mask=valid_step_mask,
                        detach_step=detach_step,
                        data_info=data_info)
        return in_batch

    def _init_metric(self) -> None:
        self.train_map = mAP(extended_summary=True, backend='faster_coco_eval')
        self.train_iou = IOU()
        self.val_map = mAP(extended_summary=True, backend='faster_coco_eval')
        self.val_iou = IOU()
        self.test_map = mAP(extended_summary=True, backend='faster_coco_eval')
        self.test_iou = IOU()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # print(f"batch_idx: {batch_idx}")
        self.model.init_beliefs(batch['initial_state'])
        opt = self.optimizers()
        self.batch_size, _, seq_len = batch['targets'].shape

        indices_win = self.process_slide_window(seq_len)

        # prepare to predict
        loss_item = 0
        loss4log: Dict[Any] = dict()  # type: ignore
        for i, indices in enumerate(indices_win):

            in_batch = self.process_batch(batch, indices)
            out = self.model.loss_and_filtering(in_batch)
            train_losses = out['loss']
            train_loss = sum([v for k, v in train_losses.items() if k.startswith('loss_')])

            loss_item += train_loss.item()
            opt.zero_grad()
            self.manual_backward(train_loss, retain_graph=True)
            self.clip_gradients(opt,
                                gradient_clip_val=self.cfg.TRAINER.gradient_clip_val,
                                gradient_clip_algorithm=self.cfg.TRAINER.gradient_clip_algorithm)
            opt.step()
            self.model._detach()

            for idx, (loss_name, loss_value) in enumerate(train_losses.items()):
                if loss_name.startswith('loss_'):
                    if loss_name in loss4log.keys():
                        loss4log[loss_name] += loss_value.item()
                    else:
                        loss4log[loss_name] = loss_value.item()

            # preds[:, :, indices] = out.detach()

        self.log('train_loss/sum_per_win', loss_item / len(indices_win), batch_size=self.batch_size)
        self.log('train_loss/sum_per_timestep', loss_item / seq_len, batch_size=self.batch_size)
        for loss_name, loss_value in loss4log.items():
            self.log(f"train_loss/{loss_name.strip('loss_')}_per_win",
                     loss_value / len(indices_win),
                     batch_size=self.batch_size)
            self.log(f"train_loss/{loss_name.strip('loss_')}_per_timestep",
                     loss_value / seq_len,
                     batch_size=self.batch_size,
                     prog_bar=True)

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        self.model.init_beliefs(batch['initial_state'])
        self.batch_size, dim_state, seq_len = batch['targets'].shape

        in_batch = self.process_batch(batch, torch.arange(seq_len))
        out = self.model.loss_and_filtering(in_batch)
        val_losses = out['loss']
        val_loss = sum([v for k, v in val_losses.items() if k.startswith('loss_')])
        self.log('val_loss/loss', val_loss, batch_size=self.batch_size)

        for loss_name, loss_value in val_losses.items():
            if loss_name.startswith('loss_'):
                self.log(f"val_loss/{loss_name.strip('loss_')}", loss_value, batch_size=self.batch_size)

        results = collect_mot_results_for_metric(out['model_output'], batch, self.trainer.datamodule.val.transforms,
                                                 self.pred_metric_mask, self.target_metric_mask)
        self.val_iou.update(results['preds'], results['tgts'])
        self.val_map.update(results['preds'], results['tgts'])

    def on_validation_epoch_end(self) -> None:

        map_res = self.val_map.compute()

        iou_res = self.val_iou.compute()
        self.log('val/AR', map_res['mar_1'], prog_bar=True, batch_size=self.batch_size)
        self.log('val/AR50', map_res['recall'][0, :, 0, :].mean(), prog_bar=False, batch_size=self.batch_size)
        self.log('val/AR75', map_res['recall'][5, :, 0, :].mean(), prog_bar=False, batch_size=self.batch_size)
        self.log('val/mAP', map_res['map'], prog_bar=True, batch_size=self.batch_size)
        self.log('val/mAP_50', map_res['map_50'], prog_bar=False, batch_size=self.batch_size)
        self.log('val/mAP_75', map_res['map_75'], prog_bar=False, batch_size=self.batch_size)
        self.log('val/IOU', iou_res['iou'], prog_bar=False, batch_size=self.batch_size)

        self.val_map.reset()
        self.val_iou.reset()

    def test_step(self, batch, batch_idx):
        self.model.init_beliefs(batch['initial_state'])
        self.batch_size, dim_state, seq_len = batch['targets'].shape
        # preds = torch.zeros(batch_size, dim_state, seq_len).to(self.device)

        in_batch = self.process_batch(batch, torch.arange(seq_len))
        out = self.model.loss_and_filtering(in_batch)
        test_losses = out['loss']
        test_loss = sum([v for k, v in test_losses.items() if k.startswith('loss_')])

        self.log('test_loss/loss', test_loss, batch_size=self.batch_size)
        for loss_name, loss_value in test_losses.items():
            if loss_name.startswith('loss_'):
                self.log(f"test_loss/{loss_name.strip('loss_')}", loss_value, batch_size=self.batch_size)

        results = collect_mot_results_for_metric(out['model_output'], batch, self.trainer.datamodule.test.transforms,
                                                 self.pred_metric_mask, self.target_metric_mask)
        self.test_iou.update(results['preds'], results['tgts'])
        self.test_map.update(results['preds'], results['tgts'])

    def on_test_epoch_end(self) -> None:
        map_res = self.test_map.compute()
        iou_res = self.test_iou.compute()

        self.log('test/AR', map_res['mar_1'], prog_bar=True, batch_size=self.batch_size)
        self.log('test/AR50', map_res['recall'][0, :, 0, :].mean(), prog_bar=False, batch_size=self.batch_size)
        self.log('test/AR75', map_res['recall'][5, :, 0, :].mean(), prog_bar=False, batch_size=self.batch_size)
        self.log('test/mAP', map_res['map'], prog_bar=True, batch_size=self.batch_size)
        self.log('test/mAP_50', map_res['map_50'], prog_bar=False, batch_size=self.batch_size)
        self.log('test/mAP_75', map_res['map_75'], prog_bar=False, batch_size=self.batch_size)
        self.log('test/IOU', iou_res['iou'], prog_bar=False, batch_size=self.batch_size)

        self.test_map.reset()
        self.test_iou.reset()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch_size, _, len = batch['targets'].shape
        self.model.init_beliefs(batch['initial_state'])
        preds = self.model.filtering(batch)
        # process

        return dict(preds=preds, batch=batch)
