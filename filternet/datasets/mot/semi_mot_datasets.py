import copy
import os.path as osp
from collections.abc import Mapping
from typing import Any, Optional, Union
import torch
from mmengine import fileio
from torch.utils.data import Dataset
from filternet.registry import DATASETS
from filternet.utils import check_nan_inf
from filternet.utils.bbox_mode import bbox_x1y1wh_to_cxcyah, bbox_x1y1wh_to_cxcywh


@DATASETS.register_module()
class SemiMOTDataset(Dataset):
    METAINFO = {'classes': ('pedestrian', 'car', 'soccer_baller', 'dancer')}

    def __init__(self,
                 ann_file: Optional[str] = '',
                 metainfo: Union[Mapping] = dict(classes=('pedestrian', )),
                 data_root: Optional[str] = '',
                 seq_len: int | None = None,
                 use_ndata: int | None = None,
                 split_drop_last: bool = False,
                 transforms: bool = True,
                 special_track_id: int | None = None,
                 test_mode: bool = False,
                 state_mode: str = 'cxcyah'):
        """_summary_

        Args:
            ann_file (Optional[str], optional): _description_. Defaults to ''.
            metainfo (Union[Mapping], optional): Specify the category to be used. \
                Defaults to dict(classes=('pedestrian', )).
            data_root (Optional[str], optional): _description_. Defaults to ''.
            seq_len (int | None, optional): seq_len for train, if None, will full length. Defaults to None.
            use_ndata (int | None, optional): use nums of data for train/test. Defaults to None.
            drop_last (bool, optional): _description_. Defaults to True.
            transforms (bool, optional): _description_. Defaults to True.
            special_id (_type_, optional): _description_. Defaults to None.
            test_mode (bool, optional): _description_. Defaults to False.
        """

        super().__init__()

        self.ann_file = ann_file
        if isinstance(ann_file, str):
            self.ann_file = [self.ann_file]

        self.metainfo = self._load_metainfo(copy.deepcopy(metainfo))

        self.data_root = data_root
        self.seq_len = seq_len
        self.test_mode = test_mode
        self.use_ndata = use_ndata

        self.transforms = transforms
        self.special_track_id = special_track_id
        self.split_drop_last = split_drop_last
        self.state_mode = state_mode

        if self.split_drop_last:
            assert self.seq_len

        self._parse_data()

    def __len__(self) -> int:
        return len(self.data_info)

    def _parse_data(self):
        self.origin_sequence = []
        self.sequence = []
        self.observations = []
        self.data_info = []

        for ann_file in self.ann_file:
            # support concat dataset
            data = fileio.load(osp.join(self.data_root, ann_file))
            for cat_name in self.metainfo['classes']:
                if cat_name not in data:
                    continue
                cur_class_origin_sequence = []
                cur_class_sequence = []
                cur_class_observations = []
                cur_class_data_info = []
                for sequence in data[cat_name]:
                    if self.seq_len is not None:
                        if len(sequence['sequence']
                               ) < self.seq_len + 1:  # +1 for init
                            continue
                    if self.special_track_id is not None:
                        if sequence['global_track_id'] != self.special_track_id:
                            continue
                    sequence['video']['global_track_id'] = sequence[
                        'global_track_id']
                    _ori_seq = torch.tensor(sequence['ori_sequence'],
                                            dtype=torch.float32).T
                    _seq = torch.tensor(sequence['sequence'],
                                        dtype=torch.float32).T
                    _obs = torch.tensor(sequence['measurement'],
                                        dtype=torch.float32).T

                    if self.seq_len is not None:
                        _ori_seq = torch.split(_ori_seq, self.seq_len + 1, 1)
                        _seq = torch.split(_seq, self.seq_len + 1, 1)
                        _obs = torch.split(_obs, self.seq_len + 1, 1)
                        cur_class_origin_sequence.extend(
                            _ori_seq[:-1] if self.split_drop_last else _ori_seq
                        )
                        cur_class_observations.extend(
                            _obs[:-1] if self.split_drop_last else _obs)

                        for i in range(
                                len(_obs[:-1]) if self.
                                split_drop_last else len(_obs)):
                            cur_class_data_info.append(sequence['video'])

                        # test noise seq and origin for target
                        # self.sequence.extend(
                        #     _seq[:-1] if self.split_drop_last else _seq)
                        # BUG
                        cur_class_sequence.extend(
                            _ori_seq[:-1] if self.split_drop_last else _ori_seq
                        )
                    else:
                        cur_class_origin_sequence.append(_ori_seq)
                        cur_class_sequence.append(_seq)
                        cur_class_observations.append(_obs)
                        cur_class_data_info.append(sequence['video'])

            self.origin_sequence.extend(
                cur_class_origin_sequence[:self.use_ndata])
            self.sequence.extend(cur_class_sequence[:self.use_ndata])
            self.observations.extend(cur_class_observations[:self.use_ndata])
            self.data_info.extend(cur_class_data_info[:self.use_ndata])
        if len(self.data_info) == 0:
            raise ValueError(f'No data found in the dataset, '
                             f'please check {self.data_root=},'
                             f'{self.ann_file=} and {self.metainfo=}')

    def __getitem__(self, index: int) -> Any:
        initial_state, targets, observations, info = self._prepare_data(index)
        try:
            check_nan_inf(initial_state, 'initial_state')
            # check_nan_inf(observations, 'observations')
        except ValueError:

            if getattr(self, 'valid_index', None) is None:
                return self.__getitem__(torch.randint(0, len(self), (1, )))
            else:
                return self.__getitem__(self.valid_index)

        if getattr(self, 'valid_index', None) is None:
            self.valid_index = index

        valid_step_mask = targets.isnan().any(dim=0).flatten()
        valid_step_mask = ~valid_step_mask

        return {
            'initial_state': initial_state,
            'targets': targets,
            'inputs': observations,
            'data_info': info,
            'valid_step_mask': valid_step_mask,
        }

    def _prepare_data(self, idx):

        data_info = copy.deepcopy(self.data_info[idx])
        data_info['box_mode'] = self.state_mode
        data_info['frame_ids'] = self.observations[idx][
            0, 1:]  # Initial frame not included

        ori_sequence = self.origin_sequence[idx][:, 1:].clone()
        initial_state = self.observations[idx][1:5, [0]]  # [[x,y, w/h, h], t]
        observations = self.observations[idx][1:5,
                                              1:]  # [[x,y, w/h, h], seq_len]
        # BUG:
        targets = self.sequence[idx][1:5, 1:]  # [[x,y, w/h, h], seq_len]

        # targets = bbox_x1y1wh_to_cxcyah(targets.T.clone()).T
        # observations = bbox_x1y1wh_to_cxcyah(observations.T.clone()).T
        # initial_state = bbox_x1y1wh_to_cxcyah(initial_state.T.clone()).T

        width = data_info['width']
        height = data_info['height']
        if self.state_mode == 'cxcyah':
            trans_func = bbox_x1y1wh_to_cxcyah
            norm_data = torch.tensor([width, height, 1, height]).reshape(-1, 1)
        elif self.state_mode == 'x1y1wh':
            trans_func = lambda x: x  # noqa: E731
            norm_data = torch.tensor([width, height, width,
                                      height]).reshape(-1, 1)
        elif self.state_mode == 'cxcywh':
            trans_func = bbox_x1y1wh_to_cxcywh
            norm_data = torch.tensor([width, height, width,
                                      height]).reshape(-1, 1)
        else:
            raise NotImplementedError

        ori_sequence[1:5, :] = trans_func(ori_sequence[1:5, :].T.clone()).T
        targets = trans_func(targets.T.clone()).T
        observations = trans_func(observations.T.clone()).T
        initial_state = trans_func(initial_state.T.clone()).T

        if self.transforms:
            initial_state = initial_state / norm_data
            observations = observations / norm_data
            targets = targets / norm_data
            ori_sequence[1:5, :] = ori_sequence[1:5, :] / norm_data
        data_info['origin_sequence'] = ori_sequence

        return initial_state, targets, observations, data_info

    @classmethod
    def _load_metainfo(cls, metainfo: Union[Mapping, None] = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (Mapping or Config, optional): Meta information dict.
                If ``metainfo`` contains existed filename, it will be
                parsed by ``list_from_file``.

        Returns:
            dict: Parsed meta information.
        """
        # avoid `cls.METAINFO` being overwritten by `metainfo`
        cls_metainfo = copy.deepcopy(cls.METAINFO)
        if metainfo is None:
            return cls_metainfo
        if not isinstance(metainfo, (Mapping)):
            raise TypeError('metainfo should be a Mapping or Config, '
                            f'but got {type(metainfo)}')

        for k, v in metainfo.items():
            cls_metainfo[k] = v
        return cls_metainfo

    def inverse_data(self, batch_data, batch):
        data_infos = batch['data_info']
        inversed_data = []
        if self.transforms:
            for data, data_info in zip(batch_data, data_infos):
                width = data_info['width']
                height = data_info['height']
                if self.state_mode == 'cxcyah':
                    norm_data = torch.tensor([width, height, 1, height
                                              ]).reshape(-1, 1).to(data.device)

                elif self.state_mode == 'cxcywh' or self.state_mode == 'x1y1wh':
                    norm_data = torch.tensor([width, height, width, height
                                              ]).reshape(-1, 1).to(data.device)
                else:
                    raise ValueError(f'Invalid state_mode: {self.state_mode}')
                inversed_data.append(data * norm_data)
            return torch.stack(inversed_data)
        else:
            return data


if __name__ == '__main__':
    data = SemiMOTDataset(
        ann_file='half-train_track_0.05noise2track.json',
        data_root='./data/MOT17',
        # metainfo=dict(classes=('pedestrian')),
    )
