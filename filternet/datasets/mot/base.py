from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import default_collate


def mot_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        data_info[i]['frame_ids'] = F.pad(data_info[i]['frame_ids'],
                                          (0, pad_len))
    return {
        'valid_step_mask': default_collate(masks),
        'targets': default_collate(padded_targets),
        'inputs': default_collate(padded_inputs),
        'initial_state': default_collate(initial_state),
        'data_info': data_info,
    }
