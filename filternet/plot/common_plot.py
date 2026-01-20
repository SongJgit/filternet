import os.path as osp
from typing import List

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from torch import Tensor

from filternet.utils import compute_metric

# cSpell: ignore Dtrack, randn


class Plot:

    def __init__(self):
        self.color = ['-ro', '-go', '-bo', '-yo', '-co']
        self.axis_names = [
            [],
            ['X'],
            ['X', 'Y'],
            ['X', 'Y', 'Z'], ]
        self.fontsize = 32
        self.figsize = (30, 17)

    def print_metrics(self, predict: Tensor, target: Tensor, name: str = None):
        scalar, curves = compute_metric(predict, target)
        print(f"{name} - MSE LOSS: {scalar['mse_dB']}[dB]")
        print(f"{name} - MSE LOSS: {scalar['mse']}")
        print(f"{name} - RMSE LOSS: {scalar['rmse']}")
        return scalar, curves

    def plot_scalar(self,
                    predict: Tensor,
                    target: Tensor,
                    save_dir: str | None = None,
                    description: str | None = None) -> None:
        scalar_metrics, _ = compute_metric(predict, target)

        label = ['MSE(dB)', 'RMSE']

        batch = predict.shape[0]
        metrics = [scalar_metrics['mse_dB'], scalar_metrics['rmse']]

        for lbl, metric in zip(label, metrics):
            plt.figure(figsize=self.figsize)
            xplt = range(0, batch)
            yplt = metric * torch.ones(batch)
            plt.plot(xplt, yplt, self.color[0], label=f'{lbl}')

            plt.xlabel('Number of Samples', fontsize=self.fontsize)
            plt.ylabel(f'{lbl} Value', fontsize=self.fontsize)
            plt.title(f'{description}: {lbl} For All Axes', fontsize=self.fontsize)
            plt.legend(fontsize=self.fontsize)
            plt.grid(True)
            plt.tick_params(labelsize=20)
            if save_dir is not None:
                plt.savefig(osp.join(save_dir, f'{description}_{lbl}_All_Step.png'))
            # plt.show()

            # plt.close()

    def plot_axis_scalar(self,
                         predict: Tensor,
                         target: Tensor,
                         save_dir: str | None = None,
                         description: str | None = None) -> None:
        """_summary_

        Args:
            predict (Tensor): [batch_size, num_axis, seq_len]
            target (Tensor): [batch_size, num_axis, seq_len]
            save_dir (str): _description_
            description (str | None, optional): train/val/test. Defaults to None.
            metric_mask (List[bool] | None, optional):compute axis, like 9 dims ca model position x,y,z = [0,3,6].
                Defaults to None.
        """
        scalar_metrics, _ = compute_metric(predict, target)

        batch = predict.shape[0]
        axis_mse_dB_scalar = scalar_metrics['axis_mse_dB']
        axis_rmse_scalar = scalar_metrics['axis_rmse']
        num_axis = predict.shape[1]

        label = ['MSE(dB)', 'RMSE']
        xplt = range(0, batch)
        for lbl, metrics in zip(label, [axis_mse_dB_scalar, axis_rmse_scalar]):
            plt.figure(figsize=self.figsize)
            for idx, name in enumerate(self.axis_names[num_axis]):
                yplt = metrics[idx] * torch.ones(batch)
                plt.plot(xplt, yplt, self.color[idx], label=f'{name}-Axis-{lbl}')

            plt.xlabel('Number of samples', fontsize=self.fontsize)
            plt.ylabel(f'{lbl} Value', fontsize=self.fontsize)
            plt.title(f'{description}: {lbl} For {num_axis} Axes', fontsize=self.fontsize)
            plt.legend(fontsize=self.fontsize)
            plt.grid(True)
            plt.tick_params(labelsize=20)
            if save_dir is not None:
                plt.savefig(osp.join(save_dir, f'{description}_{lbl}_{num_axis}_Axes_All_Step.png'))
            # plt.show()

            # plt.close()

    def plot_curves(self,
                    predict: Tensor,
                    target: Tensor,
                    save_dir: str | None = None,
                    description: str | None = None) -> None:

        _, curves_metric = compute_metric(predict, target)

        rmse_curves = curves_metric['rmse_curves']
        mse_dB_curves = curves_metric['mse_dB_curves']
        label = ['MSE(dB)', 'RMSE']
        for lbl, metrics in zip(label, [mse_dB_curves, rmse_curves]):
            plt.figure(figsize=self.figsize)
            xplt = range(predict.shape[-1])
            yplt = metrics
            plt.plot(xplt, yplt, self.color[0], label=f'{lbl}')
            plt.xlabel('Step', fontsize=self.fontsize)
            plt.ylabel(f'{lbl} Value', fontsize=self.fontsize)
            plt.title(f'{description}: {lbl} For All Axes Each Step', fontsize=self.fontsize)
            plt.legend(fontsize=self.fontsize)
            plt.grid(True)
            plt.tick_params(labelsize=20)

            if save_dir is not None:
                plt.savefig(osp.join(save_dir, f'{description}_{lbl}_Each_Step.png'))
            # plt.show()
            # plt.close()

    def plot_axis_curves(self,
                         predict: Tensor,
                         target: Tensor,
                         save_dir: str | None = None,
                         description: str | None = None) -> None:
        """_summary_

        Args:
            predict (Tensor): [batch, ...]
            target (Tensor): [batch, ...]
            save_dir (str): save plot image, like './' save to current dir.
            description (str | None, optional):  The description of this plotting.
                            Like 'train_observation' . Defaults to None.
            axis_names (List[str], optional): _description_. Defaults to ['x', 'y', 'z'].
        """

        _, curves_metric = compute_metric(predict, target)
        num_axis = predict.shape[1]
        axis_rmse_curves = curves_metric['axis_rmse_curves']
        # axis_mse_dB_curves = curves_metric['axis_mse_dB_curves']
        x = range(0, predict.shape[-1])
        # label = ['MSE(dB)', 'RMSE']
        label = ['RMSE']

        for lbl, metrics in zip(label, [axis_rmse_curves]):
            plt.figure(figsize=self.figsize)  # 16:9
            for axis, name in enumerate(self.axis_names[num_axis]):
                y = metrics[axis]
                plt.plot(x, y, self.color[axis], label=f'{name}-Axis-{lbl}')

            plt.xlabel('Step', fontsize=self.fontsize)
            plt.ylabel(f'{lbl} Value', fontsize=self.fontsize)
            plt.legend(fontsize=self.fontsize)
            plt.title(f'{description}: {lbl} For {num_axis} Axes Each Step', fontsize=self.fontsize)
            plt.grid(True)
            plt.tick_params(labelsize=20)

            if save_dir is not None:
                plt.savefig(osp.join(save_dir, f'{description}_{lbl}_{num_axis}_Axes_Each_Step.png'))
            # plt.show()
            # plt.close()


def plot_model_prob(model_probs: List[torch.Tensor] | torch.Tensor,
                    mode: str = 'single',
                    description: List[str] = ['CV', 'CA'],
                    title_prefix: str | None = None) -> None:
    """[[num_filer, seq_len], ...] or [num_filter, seq_len] or [bs, num_filter,
    seq_len]

    Args:
        model_probs (List[torch.Tensor]): _description_
        save_dir (str | None, optional): _description_. Defaults to None.
    """
    color = ['r', 'g', 'b', 'y', 'c']
    if isinstance(model_probs, torch.Tensor):
        if model_probs.ndim == 2:
            # [num_filter, seq_len]
            num_data = 1
            num_filter = model_probs.shape[0]
            model_probs = model_probs[None, ...]  # -> [bs, num_filter, seq_len]
        else:
            # [bs, num_filter, seq_len]
            num_data, num_filter = model_probs.shape[:2]
    else:
        # [[num_filer, seq_len], ...]
        num_data = len(model_probs)
        num_filter = model_probs[0].shape[0]

    if mode == 'avg':
        if not isinstance(model_probs, torch.Tensor):
            model_probs = torch.cat(model_probs, dim=0)
        model_probs = model_probs.mean(dim=0, keepdim=True)
        num_data = len(model_probs)

    plt.figure(figsize=(19, 10), dpi=600)
    fontsize = 32
    for idx in range(num_data):
        for jdx in range(num_filter):
            y = model_probs[idx][jdx].cpu().detach().numpy()
            x = torch.arange(y.shape[-1])  # seq_len
            plt.plot(x, y, color[jdx])

    legend_elements = [
        Line2D([jdx], [jdx], color=color[jdx], lw=2, label=f'{description[jdx]} Filter Prob')
        for jdx in range(num_filter)]

    plt.legend(fontsize=fontsize, handles=legend_elements)
    plt.xlabel('Step', fontsize=32)
    plt.ylabel('Probability Value', fontsize=32)
    if mode == 'avg':
        plt.title(f'{title_prefix} Model Probability Average', fontsize=32)
    else:
        plt.title(f'{title_prefix} Trajectory Model Probability Compare', fontsize=32)
    plt.legend(fontsize=fontsize, handles=legend_elements)
    plt.grid(True)
    plt.tick_params(labelsize=20)
