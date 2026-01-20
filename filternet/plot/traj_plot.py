import os.path as osp
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_2Dtrack(
    data: List[torch.Tensor] | torch.Tensor,
    mask: torch.tensor | List[str] | np.array = None,
    label: List[str] = ['X', 'Y'],
    legend: List[str | int] | None = None,
    save_dir: str | None = None,
    useMathText=False,
) -> None:
    """Along axis plot track.

    Example : data = torch.randn((1000, 3, 100)) # [batch, num_state, seq_len]
              plot_track(data,
                        plot_axis = [0, 1, 2])
                        axis_name = ['X', 'Y', 'Z'])
    Args:
        data (torch.tensor): [batch, num_state, seq_len]
        plot_axis (List[int], optional): The dimension you want to draw. Defaults to [0, 3, 6].
        axis_name (List[str], optional): The name of the dimension you want to draw, corresponding
                                        to the parameters plot_axis.
                                        Defaults to ['X', 'Y', 'Z'].
        save_dir (str | None, optional): save dir, like ../plot/. Defaults to None.
    """
    num_data = len(data)
    if mask is None:
        mask = np.ones(data[0].shape[0]).astype(bool)
    if isinstance(mask, list):
        mask = np.array(mask).astype(bool)

    # for axis, name in zip(range(plot_axis), axis_name):
    plt.figure(figsize=(6, 3), dpi=120)
    for cur_data in data:
        masked_cur_data = cur_data[mask, :]
        assert masked_cur_data.shape[0] == 2, f'2D track expect x and y, but got {masked_cur_data.shape[0]=}'
        x = masked_cur_data[0, :]
        y = masked_cur_data[1, :]
        plt.plot(x, y)
        plt.annotate('', xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]), arrowprops=dict(facecolor='red', arrowstyle='-|>'))
    if legend is not None:
        assert len(legend) == num_data, f'{len(legend)=} and {num_data=} should be the same.'
        plt.legend(legend, fontsize=20)
    else:
        pass
    plt.xlabel(label[0], fontsize=15)
    plt.ylabel(label[1], fontsize=15)
    plt.title('Track')
    plt.grid(True)
    plt.ticklabel_format(useMathText=useMathText)
    plt.tick_params(labelsize=20)

    if save_dir is not None:
        plt.savefig(osp.join(save_dir, 'track.png'))
        # plt.show()


def plot_3Dtrack(
    data: List[torch.Tensor] | torch.Tensor | np.array,
    mask: torch.tensor | List[int | str] | np.array | None = None,
    save_dir: str | None = None,
    label: List[str | int] = ['X', 'Y', 'Z'],
    legend: List[str | int] | None = None,
    useMathText=False,
) -> None:
    """_summary_

    Args:
        data (List[torch.Tensor] | torch.Tensor | np.array): [[num_state, seq_len], ...]
        mask (torch.tensor | List[int | str] | np.array): [1,1,1] for x,y,z
        save_dir (str | None, optional): _description_. Defaults to None.
        useMathText (bool, optional): _description_. Defaults to False.
    """
    num_data = len(data)
    if mask is None:
        mask = np.ones(data[0].shape[0]).astype(bool)
    if isinstance(mask, list):
        mask = np.array(mask).astype(bool)
    ax = plt.figure(figsize=(19, 10), dpi=150).add_subplot(projection='3d')
    for cur_data in data:
        masked_cur_data = cur_data[mask, :]
        assert masked_cur_data.shape[0] == 3, f'3D track expect x, y and z, but got {masked_cur_data.shape[0]=}'
        x = masked_cur_data[0, :]
        y = masked_cur_data[1, :]
        z = masked_cur_data[2, :]

        ax.plot(x, y, z)
        ax.text(x[-1], y[-1], z[-1], 'End', color='red')
    if legend is None:
        legend = [i for i in range(num_data)]
    assert len(legend) == num_data, f'{len(legend)=} and {num_data=} should be the same.'
    ax.set_xlabel(label[0], fontsize=15)
    ax.set_ylabel(label[1], fontsize=15)
    ax.set_zlabel(label[2], fontsize=15)
    plt.title('Track')
    plt.legend(legend, fontsize=15)
    plt.grid(True)
    plt.ticklabel_format(useMathText=useMathText)
    plt.tick_params(labelsize=15)
    if save_dir is not None:
        plt.savefig(osp.join(save_dir, 'track.png'))


class PlotTrajs:

    def __init__(self,
                 data: List[np.ndarray | torch.Tensor],
                 pos_axis: List[int] = [0, 3],
                 vel_axis: List[int] = [1, 4],
                 acc_axis: List[int] = [2, 5]):

        self.pos_axis = pos_axis
        self.vel_axis = vel_axis
        self.acc_axis = acc_axis

        lbl = ['x', 'y', 'z']
        self.key_lbl = dict(x=0, y=1, z=2)

        self.pos_data = []
        for _data in data:
            _pos = _data[self.pos_axis, :]
            self.pos_data.append(_pos)
        self.pos_lbl = [lbl[i] for i in range(len(self.pos_axis))]

        if vel_axis is not None:
            self.vel_data: List[np.ndarray] = [[] for i in self.vel_axis]
            for idx, ax in enumerate(self.vel_axis):
                for _data in data:
                    ind = np.arange(_data.shape[-1]).reshape(1, -1)
                    _single_vel = _data[[ax], :]
                    _single_vel = np.concatenate([ind, _data[[ax], :]], axis=0)
                    self.vel_data[idx].append(_single_vel)

            # self.vel_lbl = ['step', 'velocity']
            self.vel_lbl = [['step', 'vel_' + lbl[i]] for i in range(len(self.vel_axis))]
        if acc_axis is not None:
            self.acc_data: List[np.ndarray] = [[] for i in self.acc_axis]
            for idx, ax in enumerate(self.acc_axis):
                for _data in data:
                    ind = np.arange(_data.shape[-1]).reshape(1, -1)
                    _single_acc = _data[[ax], :]
                    _single_acc = np.concatenate([ind, _data[[ax], :]], axis=0)
                    self.acc_data[idx].append(_single_acc)
            self.acc_lbl = [['step', 'acc_' + lbl[i]] for i in range(len(self.acc_axis))]

    def __len__(self):
        return len(self.data)

    def plot_pos(self, idx=None):
        if idx is None:
            plot_2Dtrack(self.pos_data, label=self.pos_lbl)
        else:
            plot_2Dtrack([self.pos_data[idx]], label=self.pos_lbl)

    def plot_vel(self, idx=None, lbl='x'):
        if idx is None:
            plot_2Dtrack(self.vel_data[self.key_lbl[lbl]], label=self.vel_lbl[self.key_lbl[lbl]])
        else:
            plot_2Dtrack([self.vel_data[self.key_lbl[lbl]][idx]], label=self.vel_lbl[self.key_lbl[lbl]])

    def plot_acc(self, idx=None, lbl='x'):
        if idx is None:
            plot_2Dtrack(self.acc_data[self.key_lbl[lbl]], label=self.acc_lbl[self.key_lbl[lbl]])
        else:
            plot_2Dtrack([self.acc_data[self.key_lbl[lbl]][idx]], label=self.acc_lbl[self.key_lbl[lbl]])


class PlotSingleTraj:

    def __init__(self,
                 data: np.ndarray | torch.Tensor,
                 pos_axis: List[int] = [0, 3],
                 vel_axis: List[int] = [1, 4],
                 acc_axis: List[int] = [2, 5]):
        self.data = data
        self.pos_axis = pos_axis
        self.vel_axis = vel_axis
        self.acc_axis = acc_axis

        lbl = ['x', 'y', 'z']
        self.key_lbl = dict(x=0, y=1, z=2)
        ind = np.arange(self.data.shape[-1]).reshape(1, -1)

        if self.vel_axis is not None:
            self.vel_data: List[np.ndarray] = [[] for i in self.vel_axis]
            for idx, ax in enumerate(self.vel_axis):
                self.vel_data[idx].append(np.concatenate([ind, self.data[[ax], :]], axis=0))

                self.vel_lbl = [['step', 'vel_' + lbl[i]] for i in range(len(self.vel_axis))]

        if self.acc_axis is not None:
            self.acc_data: List[np.ndarray] = [[] for i in self.acc_axis]
            for idx, ax in enumerate(self.acc_axis):
                self.acc_data[idx].append(np.concatenate([ind, self.data[[ax], :]], axis=0))

            self.acc_lbl = [['step', 'acc_' + lbl[i]] for i in range(len(self.acc_axis))]

    def __len__(self):
        return self.data.shape[-1]

    def plot_pos(self):
        plot_2Dtrack([self.data[self.pos_axis, :]])

    def plot_vel(self, lbl='x'):

        plot_2Dtrack(self.vel_data[self.key_lbl[lbl]], label=self.vel_lbl[self.key_lbl[lbl]])

    def plot_acc(self, lbl='x'):
        plot_2Dtrack(self.acc_data[self.key_lbl[lbl]], label=self.acc_lbl[self.key_lbl[lbl]])
