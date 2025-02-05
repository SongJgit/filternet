from __future__ import annotations

import os
import os.path as osp
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor


def training_info() -> str:
    today = datetime.today()
    now = datetime.now()
    str_today = today.strftime('%m.%d.%y')
    str_now = now.strftime('%H:%M:%S')
    str_time = ' '.join([str_today, str_now])
    print(f'Current Time = {str_time}')
    return str_time


def generate_save_dir(root: Optional[str] = None,
                      project: str = 'project',
                      name: Optional[str] = None,
                      mode: str = 'train') -> Dict:
    if not root:
        root = os.getcwd()

    project_root = osp.join(root, project, name + '_v0')
    save_dirs: Dict[str, str] = {'experiments_dir': project_root}
    save_dirs['new_name'] = name + '_v0'

    while osp.exists(project_root):
        suffix = int(project_root.split('_v')[-1]) + 1
        prefix = project_root.split('_')[:-1]
        prefix.append(f'v{suffix}')
        project_root = '_'.join(prefix)
        save_dirs['new_name'] = name + f'_v{suffix}'

    save_dirs['weight_dir'] = osp.join(project_root,
                                       'checkpoints')  # type: ignore

    save_dirs['config_dir'] = osp.join(project_root, 'configs')  # type: ignore

    save_dirs['eval_dir'] = osp.join(project_root,
                                     'eval_results')  # type: ignore

    save_dirs['log_images_dir'] = osp.join(project_root,
                                           'log_images')  # type: ignore
    save_dirs['log_metrics_dir'] = osp.join(project_root, 'log_metrics')

    # eval_sub_dir = osp.join(eval_dir, 'imgs')

    # if mode == 'train':
    #     for key, dir in save_dirs.items():
    #         if not osp.exists(dir) and key.endswith('dir'):
    #             os.makedirs(dir)

    # save_dir["eval_sub_dir"] = eval_sub_dir
    return save_dirs


def get_img(path: str):
    suffix = ['jpeg', 'jpg', 'png']
    file_list = os.listdir(path)
    img_list = [img for img in file_list if img.split('.')[-1] in suffix]
    prefix_list = [img.split('.')[0] for img in img_list]
    img_list = [osp.join(path, img) for img in file_list]
    return prefix_list, img_list


def metrics2df(metrics: List[Dict[str, Any]],
               axis_name: List[str] = ['X', 'Y', 'Z'],
               data_from: List[str] = ['Obs_Error'],
               save_dir: str | None = None) -> pd.DataFrame:
    """convert metrics to pandas for wandb logger.

    Example:
        num_axis = 3
        metrics = [{'mse': torch.tensor(50.0), 'axis_mse': [torch.randn(1) for _ in range(num_axis)]}]
        df = metrics2df(metrics, save_dir= './')

    Args:
        metrics (List[Dict[str, Any]]): from utils.compute_metric output
        axis_name (List[str], optional): num metrics axis. Defaults to ['X', 'Y', 'Z'].
        data_from (List[str], optional): Metric from, like val KF out, val obs error. Defaults to ['Obs_Error'].
        save_dir (str | None, optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    df_list = []
    for metric, data in zip(metrics, data_from):
        col = ['MetricFrom']
        row = [data]
        for key, val in metric.items():
            if isinstance(val, list):
                for name, v in zip(axis_name, val):
                    col.append(f'{name}_{key}')
                    row.append(v.item())
            else:
                col.append(key)
                row.append(val.item())
        temp = {key: val for key, val in zip(col, row)}
        df_list.append(temp)
    df = pd.DataFrame(df_list)
    if save_dir is not None:
        df.to_csv(osp.join(save_dir, 'Metric.csv'), index=False)
    return df


def check_nan_inf(tensor: Tensor, name: str) -> None:
    if torch.any(torch.isnan(tensor)):
        raise ValueError(f'{name} has nan, {tensor}')
    if torch.any(torch.isinf(tensor)):
        raise ValueError(f'{name} has inf, {tensor}')


def get_path_ckpt_config(root_path: str, mode: str = 'min') -> Tuple[str, str]:
    """Get the config.py and weights in the specified directory. The config.py
    file is assumed to be in the root_path/configs directory. The checkpoint
    file must named in the following format,`={metric value}}.ckpt` to make it
    easier to select ckpt based on the metric's mode(min or max is better).\
    Like Error metric is better to be minimized, so the mode should be 'min'.

        root_path/
        │
        ├── configs/
        │   └── config.py
        │
        └── checkpoints/
            ├── xxx_metric_name=xxx.ckpt
            └── xxx_metric_name=yyy.ckpt

    Args:
        root_path (str): _description_
        mode (str): metric mode, min or max

    Returns:
        _type_: _description_
    """

    path2config = os.path.join(root_path, 'configs', 'config.py')
    ckpt_folder = os.path.join(root_path, 'checkpoints')
    path2ckpt = get_ckpt(ckpt_folder, mode)

    return path2ckpt, path2config


def get_ckpt(ckpt_folder: str, mode: str = 'min') -> str:
    mode = 'min' if not isinstance(mode, str) else mode

    ckpt_arr = np.array(os.listdir(ckpt_folder))
    metrics = np.array(
        [float(ckpt.split('=')[-1].strip('.ckpt')) for ckpt in ckpt_arr])

    valid_indices = np.where(~np.isnan(metrics) & ~np.isinf(metrics))[0]

    if len(valid_indices) == 0:
        raise ValueError(f'No valid checkpoint found in {ckpt_folder}')

    valid_metrics = metrics[valid_indices]
    valid_ckpt = ckpt_arr[valid_indices]

    if mode.lower() == 'min':
        best_ckpt_idx = np.argmin(valid_metrics)
    elif mode.lower() == 'max':
        best_ckpt_idx = np.argmax(valid_metrics)
    else:
        raise ValueError(f"mode should be 'min' or 'max', but got {mode}")
    path2ckpt = os.path.join(ckpt_folder, valid_ckpt[best_ckpt_idx])
    return path2ckpt


def _safe_divide(
        num: Union[torch.Tensor, np.ndarray],
        denom: Union[torch.Tensor, np.ndarray],
        zero_division: float = 0.0) -> Union[torch.Tensor, np.ndarray]:
    """Safe division, by preventing division by zero.

    Args:
        num (Union[Tensor, np.ndarray]): _description_
        denom (Union[Tensor, np.ndarray]): _description_
        zero_division (float, optional): _description_. Defaults to 0.0.

    Returns:
        Union[Tensor, np.ndarray]: Division results.
    """
    if isinstance(num, np.ndarray):
        num = num if np.issubdtype(num.dtype,
                                   np.floating) else num.astype(float)
        denom = denom if np.issubdtype(denom.dtype,
                                       np.floating) else denom.astype(float)
        results = np.divide(
            num,
            denom,
            where=(denom != np.array([zero_division]).astype(float)))
    else:
        num = num if num.is_floating_point() else num.float()
        denom = denom if denom.is_floating_point() else denom.float()
        zero_division = torch.tensor(zero_division).float().to(denom.device)
        results = torch.where(denom != 0, num / denom, zero_division)
    return results


def expand_dim(x: torch.Tensor) -> torch.Tensor:
    # [batch_size, m] -> [1, batch_size, m]
    batch_size = x.shape[0]
    expanded = torch.empty(1, batch_size, x.shape[-1]).to(x.device)
    expanded[0, :, :] = x
    return expanded
