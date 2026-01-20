from __future__ import annotations

import os
import os.path as osp
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from importlib.util import find_spec
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
# from .logger import logger
from torch import Tensor
import logging
import copy
from mmengine import Config
from filternet.registry import MODELS

logger = logging.getLogger(__name__)


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
    """Generate directory to save exp.

    Args:
        root (Optional[str], optional): Root directory. Defaults to None.
        project (str, optional): Project Name, Crossponding with the Wandb/SwanLab Project name. Defaults to 'project'.
        name (Optional[str], optional): Experiment name.
        mode (str, optional): _description_. Defaults to 'train'.

    Returns:
        Dict: _description_
    """
    if not root:
        root = os.getcwd()

    project_root = osp.join(root, project, name + '_v0')
    save_dirs: Dict[str, str] = {'exp_dir': project_root}
    save_dirs['exp_name'] = name + '_v0'

    while osp.exists(project_root) and mode == 'train':
        suffix = int(project_root.split('_v')[-1]) + 1
        prefix = project_root.split('_')[:-1]
        prefix.append(f'v{suffix}')
        project_root = '_'.join(prefix)
        save_dirs['exp_name'] = name + f'_v{suffix}'

    save_dirs['exp_dir'] = project_root

    save_dirs['weight_dir'] = osp.join(project_root, 'checkpoints')  # type: ignore

    save_dirs['config_dir'] = osp.join(project_root, 'configs')  # type: ignore

    save_dirs['eval_dir'] = osp.join(project_root, 'eval_results')  # type: ignore

    save_dirs['log_images_dir'] = osp.join(project_root, 'log_images')  # type: ignore
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


def get_path_ckpt_config(root_path: str, mode: str = 'min'):
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

    if os.path.isdir(root_path):
        path2config = os.path.join(root_path, 'configs', 'config.py')
        ckpt_folder = os.path.join(root_path, 'checkpoints')
        path2ckpt = get_ckpt(ckpt_folder, mode)
    elif os.path.splitext(root_path)[1] in ('.ckpt', '.pt', '.pth'):
        path2config = os.path.join(os.path.dirname(os.path.dirname(root_path)), 'configs', 'config.py')
        path2ckpt = root_path
    return path2ckpt, path2config


def get_ckpt(ckpt_folder: str, mode: str = 'min'):
    mode = 'min' if not isinstance(mode, str) else mode

    ckpt_arr = np.array(os.listdir(ckpt_folder))
    metrics = np.array([float(ckpt.split('=')[-1].strip('.ckpt')) for ckpt in ckpt_arr])

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


def _safe_divide(num: Union[torch.Tensor, np.ndarray],
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
        num = num if np.issubdtype(num.dtype, np.floating) else num.astype(float)
        denom = denom if np.issubdtype(denom.dtype, np.floating) else denom.astype(float)
        results = np.divide(num, denom, where=(denom != np.array([zero_division]).astype(float)))
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


def package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    >>> package_available('os')
    True
    >>> package_available('bla')
    False
    """
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


def dB_to_lin(x: Union[torch.Tensor, np.ndarray, list, float, int]) -> Union[torch.Tensor, np.ndarray]:
    return 10 ** (x / 10)


def lin_to_dB(x: Union[torch.Tensor, np.ndarray, list, float, int]) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(x, (list, float, int)):
        x = np.array(x)
    assert not (x <= 0).any(), 'x must be positive'
    return 10 * np.log10(x)


def dummy_input(model_name: str | None, dim_state: int, dim_obs: int, device: torch.device) -> Dict:
    """Generate dummy input for model.

    Args:
        model_name (str | None): _description_
        dim_state (int): _description_
        dim_obs (int): _description_
        device (torch.device): _description_

    Returns:
        Dict: _description_
    """
    if 'fusion' in model_name.lower():
        init_state = torch.randn(2, dim_state, 1).to(device)  # [BS, M, 1]
        input = torch.randn(2, 4, 1).to(device)
        correction = torch.randn(2, 2, 1).to(device)
        input = [input, correction]

    elif 'mot' in model_name.lower():
        input = [torch.randn(2, dim_obs, 1).to(device)]
        init_state = torch.randn(2, 4, 1).to(device)

    else:
        # common
        input = [torch.randn(2, dim_obs, 1).to(device)]
        init_state = torch.randn(2, dim_state, 1).to(device)

    return dict(init=init_state, input=input)


def model_summary(model, depth=5):
    model_name = model.__class__.__name__
    try:
        from torchinfo import summary
    except Exception as e:
        logger.info(e)
        logger.info("Please install torchinfo use 'pip install torchinfo'")

    try:
        model = copy.deepcopy(model)
        device = next(model.parameters()).device
        dummy = dummy_input(model_name, model.dim_state, model.dim_obs, device)
        model.init_beliefs(dummy['init'])

        logger.info('Model summary from torchinfo:')
        table = summary(model,
                        input_data=dummy['input'],
                        depth=depth,
                        col_names=['input_size', 'output_size', 'num_params', 'mult_adds'])
        return table
    except Exception as e:
        logger.warning(e)
        logger.warning('torchinfo is not successfully executed!')


def model_grad_graph(model, save_dir: str = None):
    model_name = model.__class__.__name__
    try:
        from torchviz import make_dot
    except Exception as e:
        logger.info(e)
        logger.info("Please 'pip install torchviz' and sudo apt-get install graphviz.")
    try:

        device = next(model.parameters()).device
        dummy = dummy_input(model_name, model.dim_state, model.dim_obs, device)
        model.init_beliefs(dummy['init'])

        y = model(*dummy['input'])
        dot = make_dot(y, params=dict(model.named_parameters()))
        dot.render('model_grad_graph', directory=save_dir, format='png', cleanup=True)
        logger.info(f'Model_grad_graph saved at {osp.join(save_dir, "model_grad_graph.png")}.')
    except Exception as e:
        logger.warning(e)
        logger.warning('Generated model grad graph failed!')


def load_state_dict_from_pl(state_dict: Dict | OrderedDict, model: torch.nn.Module) -> torch.nn.Module:
    """Due to the use of lightning, there is an extra layer of encapsulation
    that needs to be peeled off.

    Args:
        state_dict (Dict | OrderedDict): lightning.checkpoints or state_dict
        model (torch.nn.Module): _description_

    Returns:
        torch.nn.Module: _description_
    """

    if not isinstance(state_dict, OrderedDict):
        state_dict = state_dict['state_dict']

    load_success = False
    for name, param in model.named_parameters():

        if 'model.' + name in state_dict:
            param.data = state_dict['model.' + name]
            load_success = True
        else:
            load_success = False
            # print(f'{name} not exist in state_dict')

    if not load_success:
        logger.info('Trying to load model without prefix "model." ')
        for name, param in model.named_parameters():

            if name in state_dict:
                param.data = state_dict[name]
                load_success = True
            else:
                load_success = False
                # print(f'{name} not exist in state_dict')
    if load_success:
        logger.info('****' * 20)
        logger.info('Loading successfully')
        logger.info('****' * 20)
    return model


def attempt_load_model(ckpt_path, mode='max'):

    if os.path.splitext(ckpt_path)[1] in ('.ckpt', '.pt', '.pth'):
        logger.info(f'Load best model from {ckpt_path}')
        ckpt = torch.load(ckpt_path)
        if 'cfg' in ckpt:
            cfg = Config(ckpt['cfg'])
        else:
            cfg = Config.fromfile(os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), 'configs', 'config.py'))
    else:
        ckpt_path, config_path = get_path_ckpt_config(ckpt_path, mode)
        logger.info(f'Load best model from {ckpt_path}')
        ckpt = torch.load(ckpt_path)
        if 'cfg' in ckpt.keys():
            cfg = Config(ckpt['cfg'])
        else:
            cfg = Config.fromfile(config_path)

    model = MODELS.build(cfg.MODEL)
    model = load_state_dict_from_pl(ckpt, model)

    return model
