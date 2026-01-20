from typing import Tuple, Dict
from torch import Tensor
from filternet.utils import MSE
import torch
from filternet.utils import logger


def collect_fusion_results(predictions: Tuple) -> Dict[str, Dict[str, Tensor]]:
    collect = {}
    for preds in predictions:
        # every batch
        batch_date: list = preds['data_date']
        for idx, date in enumerate(batch_date):
            if date not in collect:
                collect[date] = {}
                collect[date]['preds'] = [preds['preds'][idx]]
                collect[date]['targets'] = [preds['targets'][idx]]
                # collect[date]['ground_truth'] = [preds['ground_truth'][idx]]
                collect[date]['gps'] = [preds['gps'][idx]]
                collect[date]['wheel'] = [preds['wheel'][idx]]
                collect[date]['valid_step_mask'] = [preds['valid_step_mask'][idx]]
                collect[date]['sub_id'] = [preds['sub_id'][idx]]
            else:
                collect[date]['preds'].append(preds['preds'][idx])
                collect[date]['targets'].append(preds['targets'][idx])
                collect[date]['ground_truth'].append(preds['ground_truth'][idx])
                collect[date]['gps'].append(preds['gps'][idx])
                collect[date]['wheel'].append(preds['wheel'][idx])
                collect[date]['valid_step_mask'].append(preds['valid_step_mask'][idx])
                collect[date]['sub_id'].append(preds['sub_id'][idx])
    return collect


def compute_fusion_metric(collect_results: Tuple) -> Dict[str, Dict[str, Tensor]]:

    rmse4all = MSE(squared=False)
    rmse4single = MSE(squared=False)
    rmse4single_gps = MSE(squared=False)
    rmse4all_gps = MSE(squared=False)
    all_rmse = []
    # all_gt = []
    # all_pred = []
    # gps_rmse = []
    all_gps = []
    for time in collect_results.keys():
        est = torch.hstack(collect_results[time]['preds'])
        mask = torch.hstack(collect_results[time]['valid_step_mask'])
        gt = torch.hstack(collect_results[time]['targets'])
        gps = torch.hstack(collect_results[time]['gps'])
        # metric = torch.nn.MSELoss(reduction='mean')
        pred = est[..., mask]
        gt = gt[..., mask]
        gps = gps[..., mask]
        rmse4all.update(gt, pred)
        rmse = rmse4single(gt, pred)
        rmse4all_gps.update(gt, gps)
        all_rmse.append(rmse)
        logger.info(f'{time} track_len= {len(gt[0,:])}, RMSE = {rmse},  gps_rmse ={rmse4single_gps(gt,gps)}')
        all_gps.append(rmse4single_gps(gt, gps))
    logger.info(f'avg rmse = {sum(all_rmse)/len(collect_results.keys())}')
    logger.info(f'rmse4all = {rmse4all.compute()}')
    logger.info(f'avg_gps = {sum(all_gps)/len(collect_results.keys())}')
    logger.info(f'rmse4all_gps = {rmse4all_gps.compute()}')
