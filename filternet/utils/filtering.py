from typing import Callable, List

import torch
from torch import Tensor

from .metrics import MSE


def run_filter(loader, model: Callable, pred_metric_mask: List[bool] | Tensor, tgt_metric_mask: List[bool] | Tensor):
    # loader = data_module.test_dataloader()
    rmse_fn = MSE(squared=False)
    single_rmse_fn = MSE(squared=False)
    single_rmse = []
    targets = []
    results = []
    inputs = []
    batches = []
    for batch in loader:
        model.init_beliefs(batch['initial_state'])
        # print(batch['initial_state'])

        res = model.forward_loop(batch['inputs']).detach().cpu()
        bs, dim_state, _ = res.shape
        valid_step_mask = batch['valid_step_mask']

        for i in range(bs):
            masked_res = res[i, ...][pred_metric_mask, :]
            masked_targets = batch['targets'][i, ...].cpu()[tgt_metric_mask, :]

            masked_res = torch.masked_select(masked_res, valid_step_mask[i][None, ...]).reshape(masked_res.shape[0], -1)
            masked_targets = torch.masked_select(masked_targets,
                                                 valid_step_mask[i][None, ...]).reshape(masked_targets.shape[0], -1)

            masked_inputs = torch.masked_select(batch['inputs'][[i], ...].cpu(),
                                                valid_step_mask[i][None, ...]).reshape(batch['inputs'].shape[1], -1)
            targets.append(masked_targets)
            results.append(masked_res)
            inputs.append(masked_inputs)
            batch['data_info'][i]['valid_step_mask'] = valid_step_mask[i]
            batches.append(batch['data_info'][i])
            single_rmse_fn.update(masked_res, masked_targets)
            _s_rmse = single_rmse_fn.compute()
            single_rmse.append(_s_rmse)
            single_rmse_fn.reset()
            rmse_fn.update(masked_res, masked_targets)
    rmse = rmse_fn.compute()
    rmse_fn.reset()

    return results, targets, inputs, rmse, torch.tensor(single_rmse), batches
