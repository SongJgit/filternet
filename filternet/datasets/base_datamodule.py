from typing import Any, Dict, List

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

from filternet.registry import DATASETS


def common_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    targets = [data['targets'] for data in batch]
    inputs = [data['inputs'] for data in batch]
    initial_state = [data['initial_state'] for data in batch]
    max_length = max([val.shape[-1] for val in targets])
    data_info = [data['data_info'] for data in batch]
    valid_step_mask = [
        data['valid_step_mask'] if data.get('valid_step_mask') is not None else
        torch.ones(data['targets'].shape[-1]) for data in batch
    ]

    masks = []
    padded_targets = []
    padded_inputs = []
    for i in range(len(targets)):
        target = targets[i]
        input = inputs[i]
        step_mask = valid_step_mask[i]
        pad_len = max_length - target.shape[-1]
        padded_targets.append(F.pad(target, (0, pad_len)))
        padded_inputs.append(F.pad(input, (0, pad_len)))
        mask = torch.ones(target.shape[-1])
        mask = F.pad(mask, (0, pad_len))
        step_mask = F.pad(step_mask, (0, pad_len))
        mask = mask.bool() & step_mask.bool()
        masks.append(mask)
    return {
        'valid_step_mask': default_collate(masks),
        'targets': default_collate(padded_targets),
        'inputs': default_collate(padded_inputs),
        'initial_state': default_collate(initial_state),
        'data_info': data_info,
    }


class CommonDataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        # self.data_dir = cfg.data.data_path
        self.cfg = cfg

    def setup(self, stage=None) -> None:
        if stage == 'fit' or stage is None:
            self.train = DATASETS.build(self.cfg.DATA.train_dataset)
            self.val = DATASETS.build(self.cfg.DATA.val_dataset)
            self.test = DATASETS.build(self.cfg.DATA.test_dataset)
            print('train set size:', len(self.train))
            print('val set size:', len(self.val))
            print('test set size:', len(self.test))

        if stage == 'test':
            self.test = DATASETS.build(self.cfg.DATA.test_dataset)
            print('test set size:', len(self.test))
        if stage == 'predict':
            self.predict = DATASETS.build(self.cfg.DATA.test_dataset)
            print('test set size:', len(self.test))

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.cfg.DATA.train_batch_size,
                          num_workers=self.cfg.DATA.num_workers,
                          drop_last=False,
                          shuffle=self.cfg.DATA.shuffle,
                          pin_memory=self.cfg.DATA.pin_memory
                          if self.cfg.DATA.get('pin_memory') else False,
                          collate_fn=self.cfg.DATA.collate_fn if
                          self.cfg.DATA.get('collate_fn') else common_collate)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.cfg.DATA.val_batch_size,
                          num_workers=self.cfg.DATA.num_workers,
                          drop_last=False,
                          pin_memory=self.cfg.DATA.pin_memory
                          if self.cfg.DATA.get('pin_memory') else False,
                          collate_fn=self.cfg.DATA.collate_fn if
                          self.cfg.DATA.get('collate_fn') else common_collate)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.cfg.DATA.test_batch_size,
                          num_workers=self.cfg.DATA.num_workers,
                          drop_last=False,
                          pin_memory=self.cfg.DATA.pin_memory
                          if self.cfg.DATA.get('pin_memory') else False,
                          collate_fn=self.cfg.DATA.collate_fn if
                          self.cfg.DATA.get('collate_fn') else common_collate)

    def predict_dataloader(self):
        return DataLoader(self.predict,
                          batch_size=self.cfg.DATA.test_batch_size,
                          num_workers=self.cfg.DATA.num_workers,
                          drop_last=False,
                          pin_memory=self.cfg.DATA.pin_memory
                          if self.cfg.DATA.get('pin_memory') else False,
                          collate_fn=self.cfg.DATA.collate_fn if
                          self.cfg.DATA.get('collate_fn') else common_collate)
