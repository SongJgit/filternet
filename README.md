<h1 align="center">FilterNet - Learning-Aided Filtering in Python</h1>
<h3 align="center">Welcome to FilterNet</h3>
<p align="center">
<a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" alt="pre-commit" style="max-width:100%;"></a>
<a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" style="max-width:100%;"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
  <a href=""><img alt="powered by Pytorch" src="https://img.shields.io/badge/PyTorch-v1.8+-E97040?logo=pytorch&logoColor=white"></a>
</p>

## Notice

We will upload all the code once the paper submitted to "Information Fusion" has been accepted.

You can view papers related to Learning-Aided Filtering through the following [links](https://github.com/SongJgit/awesome-learning-aided-filter-papers).

## 🥳 What's New

- Feb. 2025: 🌟🌟🌟🌟🌟 Add NCLT Fusion task benchmark (with WandB logger), Lorenz Attractor benchmark (with WandB logger).
- Feb. 2025: 🌟🌟🌟 First commit. Added support for model-based Kalman filter, Extended Kalman filter, Interacting Multiple model, and learning-aided Kalman filtering KalmanNet, Split-KalmanNet, DANSE.

## Introduction

This library provides Learning-Aided/Data-Driven Kalman filtering and related optimal and non-optimal filtering software in Python.
It contains Kalman filters, Extended Kalman filters, KalmanNet, Split-KalmanNet, and Ours Semantic-Independent KalmanNet(submitted to Information Fusion, waiting review).
This library is implemented with **[Pytorch-Lightning](https://lightning.ai/docs/pytorch/stable/)**, **[MMEngine](https://github.com/open-mmlab/mmengine)**, and **[WandB](https://wandb.ai/site)**.

## Highlights

### Learning-Aided Kalman Filtering

- **Unified data structure**

  Now that Learning-Aided Kalman Filtering paper implementations have their own characteristics, which makes comparing algorithms very difficult, we use a unified data structure for the supported algorithms here, so that the user only needs to change the Datasets to seamlessly compare the algorithms.

- **Multiple tasks supported**

  Facilitates users to compare the performance of your own algorithms on different tasks, such as Lorenz Attractor, NCLT Fusion task, NCLT Estimation, and Motion Estimation, etc.

- **Easy to develop your own models**

  Many basic modules have been implemented, e.g. CV, CA modeling, etc., which can be easily extended to your own models.

- **Support for multiple GPUs and Batches**

  The code supports multi-GPU as well as mini-batch training (not supported by earlier versions of many papers, e.g. KalmanNet and DANSE).

### Advanced Features

- **[Pytorch-Lightning](https://lightning.ai/docs/pytorch/stable/)**

  We use [Pytorch-Lightning](https://lightning.ai/docs/pytorch/stable/) to simplify the training process.
  It provides a rich API that saves a lot of time in writing engineering code, such as DDP, logger, loop, etc.

- **[MMEngine](https://github.com/open-mmlab/mmengine)**

  We use MMEngine.Config to manage the model's config.
  There are several benefits of using `config file` to manage the training of the model:

  - Backup & Restore: Avoiding internal code modifications and improving the reproducibility of experiments.
  - Flexible: The `config file` provides a fast and flexible way to modify the training hyperparameter.
  - Friendliness: `config file` are separated from the model/training code, and by reading the `config file`, the user can quickly understand the hyperparameter of different models as well as the training strategies, such as optimizer, scheduler and data augmentation.

- **[WandB](https://wandb.ai/site)**

  We use [WandB](https://wandb.ai/site) to visualize the training log.
  **Pytorch-Lightning** supports a variety of loggers, such as **tensorboard** and **wandb**, but in this project, we use **wandb** as the default logger because it is very easy to share training logs, as well as very easy for multiple people to collaborate.
  In the future, we will share the logs of all models in **wandb**, so that you can easily view and compare the performance and convergence speed of different models.

-

## Model Zoo

### Learning-Aided Kalman Filtering

<div align="center">
  <b>Overview</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>Supported methods</b>
      </td>
      <td>
        <b>Supported datasets</b>
      </td>
      <td>
        <b>Supported Tasks</b>
      </td>
      <td>
        <b>Others</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li><a href="https://ieeexplore.ieee.org/document/9733186">KalmanNet (ICASSP'2021, TSP'2022)</a></li>
          <li><a href="https://ieeexplore.ieee.org/abstract/document/10120968">Split-KalmanNet (TVT'2023)</a></li>
          <li><a href="https://ieeexplore.ieee.org/document/10289946">DANSE (EUSIPCO'2023, TSP'2024)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="https://ieeexplore.ieee.org/document/9733186">Lorenz</a></li>
          <li><a href="http://journals.sagepub.com/doi/10.1177/0278364915614638">NCLT </a></li>
          <li><a href="">MOT17/MOT20/DanceTrack/SoccerNet For Motion Estimation </a></li>
        </a></li>
        </ul>
      </td>
            <td>
        <ul>
          <li><a href="https://ieeexplore.ieee.org/document/9733186">State Estimation</a></li>
          <li><a href="https://ieeexplore.ieee.org/document/10605082">Sensor Fusion</a></li>
          <li><a href="">Motion Estimation</a></li>
        </a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><b>Supported Loss</b></li>
        <ul>
          <li><a href="">MSELoss</a></li>
          <li><a href="">SmoothL1Loss</a></li>
          <li><a href="https://ieeexplore.ieee.org/document/10485649/">DANSELoss</a></li>
          <li><a href="">Any Pytorch Loss Function For Regression</a></li>
        </ul>
        </ul>
                <ul>
          <li><b>Supported Training Strategy</b></li>
        <ul>
          <li><a href="http://ieeexplore.ieee.org/document/58337/">Standard BPTT</a></li>
          <li><a href="https://ieeexplore.ieee.org/document/10605082">Alternative TBPTT</a></li>
        </ul>
        </ul>
      </td>
  </tbody>
</table>

### Model-Based Kalman Filtering

<div align="center">
  <b>Overview</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>Supported methods</b>
      </td>
      <td>
        <b>Supported datasets</b>
      </td>
      <td>
        <b>Supported Tasks</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li><a href="https://ieeexplore.ieee.org/abstract/document/5311910">Kalman filter </a></li>
          <li><a href="https://ieeexplore.ieee.org/document/1102206">Extended Kalman filter </a></li>
          <li><a href="https://ieeexplore.ieee.org/document/1299">Interacting Multiple Model </a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="https://ieeexplore.ieee.org/document/9733186">Lorenz</a></li>
          <li><a href="http://journals.sagepub.com/doi/10.1177/0278364915614638">NCLT </a></li>
          <li><a href="">MOT17/MOT20/DanceTrack/SoccerNet For Motion Estimation </a></li>
        </a></li>
        </ul>
      </td>
            <td>
        <ul>
          <li><a href="https://ieeexplore.ieee.org/document/9733186">State Estimation</a></li>
          <li><a href="https://ieeexplore.ieee.org/document/10605082">Sensor Fusion</a></li>
          <li><a href="">Motion Estimation</a></li>
        </a></li>
        </ul>
      </td>
  </tbody>
</table>

## Abbrv

| **Method** |         **Abbrv Name**         |
| :--------: | :----------------------------: |
|  **KNet**  |           KalmanNet            |
| **SKNet**  |        Split-KalmanNet         |
| **DANSE**  |             DANSE              |
| **SIKNet** | Semantic-Independent KalmanNet |

## Supervised Learning or Unsupervised Learning?

| **Methods** | **Supervised Learning** | **Unsupervised Learning** |
| :---------: | :---------------------: | :-----------------------: |
|  **KNet**   |            ✔            |             ✔             |
|  **SKNet**  |            ✔            |             ✔             |
|  **DANSE**  |            ✔            |             ✔             |
| **SIKNet**  |            ✔            |             ✔             |

## BenchMark

### ✨Note

- The number of parameters of the **same model** is not fixed, this is because the number of parameters in the network is often related to the dimensions of the **system state** and the **observation**, and the dimensions of these two are often different for different tasks. Therefore, the number of parameters of the **same model** may vary greatly for different tasks.
- 🚩🚩Such of these model are extremely sensitive to numerical values, and different machines/parameters may cause drastic changes in performance **(It is possible that the metric are slightly lower or slightly higher than in the original paper)**. **We provide the best possible metrics for each model.**

### Motion Estimation in MOT Datasets

|  Methods   | Recall@50 | Recall@75 | Recall@50:95 |
| :--------: | :-------: | :-------: | :----------: |
|   **KF**   |           |           |              |
|  **KNet**  |           |           |              |
| **SKNet**  |           |           |              |
| **SIKNet** |           |           |              |

### Lorenz Attractor

- For convenience, we directly use RMSE.
- The default parameters $q^2 = 1e-4$, and $r^2 \in \{1, 10, 100, 1000\}$.
- System state dimension: $m = 3$, observation dimension: $n = 3$.
- [More details](configs/lorenz/README.md).
- [WandB Logger](https://wandb.ai/songj/Lorenz_benchmark). It needs to be viewed in groups。

Note: In order to compare with other models, DANSE is trained using a supervised method from the source code.

|    Methods    | Params | RMSE@1  | RMSE@10 | RMSE@100 | RMSE@1000 |                 Config                 |
| :-----------: | :----: | :-----: | :-----: | :------: | :-------: | :------------------------------------: |
| **Obs Error** |  None  |  2.31   |  3.78   |  10.26   |   31.56   |                  None                  |
|   **KNet**    | 366 K  | 0.60431 | 1.18635 |  2.8958  |    Nan    | [config](configs/lorenz/knet_arch2.py) |
|   **SKNet**   | 149 K  | 0.39873 | 0.91222 | 2.01605  |  5.37797  |   [config](configs/lorenz/sknet.py)    |
|   **DANSE**   | 4.3 K  | 0.59011 | 1.20016 | 3.11831  |  7.99238  |   [config](configs/lorenz/danse.py)    |
|  **SIKNet**   | 140 K  | 0.49095 | 0.82325 | 2.03247  |  5.28136  |   [config](configs/lorenz/siknet.py)   |

### NCLT Sensor Fusion

- This implementation is sources from [Practical Implementation of KalmanNet for Accurate Data Fusion in Integrated Navigation](https://ieeexplore.ieee.org/document/10605082). If it's helpful, please cite the [paper](#anchor1).

- [More Details](configs/nclt_fusion/README.md).

- [WandB Logger](https://wandb.ai/songj/NCLT_Fusion_Benchmark?nw=nwusersongj).

- $\mathrm{Avg}= Average(\mathrm{RMSE}\text{@}\mathrm{Traj1}, \mathrm{RMSE}\text{@}\mathrm{Traj2}, \cdots, \mathrm{RMSE}\text{@}\mathrm{Traj3})$.

- $\mathrm{All} = RMSE(\mathrm{Traj1}, \mathrm{Traj2}, \cdots, \mathrm{TrajN})$.

- G-O means GPS-Only, W-O means Wheel-Only, and W-G means Wheel-GPS.

<div align="center">
  <b>RMSE in Meters for Different Methods on Test Dataset</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td colspan="2" align="center" valign="center">
        <b>Traj.Date</b>
      </td>
      <td>
        <b>2012-11-04</b>
      </td>
      <td>
        <b>2012-11-16</b>
      </td>
      <td>
        <b>2013-04-05</b>
      </td>
      <td>
        <b>Avg</b>
      </td>
      <td>
        <b>All</b>
      </td>
	    <td>
        <b>Config</b>
      </td>
    </tr>
    <tr align="center" valign="center">
      <td rowspan="2">
        <b>Methods</b>
      </td>
      <td>
        <b>Traj.Len[S]</b>
      </td>
      <td rowspan="2">
        4834
      </td>
      <td rowspan="2">
        4917
      </td>
      <td rowspan="2">
        4182
      </td>
      <td rowspan="2">
        -
      </td>
      <td rowspan="2">
        -
      </td>
      <td rowspan="2">
        -
      </td>
    </tr>
    <tr align="center" valign="center", rowspan="2">
      <td>
        <b>Params</b>
      </td>
    </tr>
    <tr align="center" valign="center">
      <td>
        <b>G-O</b>
      </td>
      <td>
        -
		</td>
      <td>
        46.141
		</td>
      <td>
        19.500
		</td>
      <td>
        34.538
		</td>
      <td>
        33.393
		</td>
      <td>
        35.084
		</td>
      <td>
        -
		</td>
	</tr>
    <tr align="center" valign="center">
      <td>
        <b>W-O EKF</b>
      </td>
      <td>
        -
		</td>
      <td>
        117.76
		</td>
      <td>
        81.61
		</td>
      <td>
        83.99
		</td>
      <td>
        94.46
		</td>
      <td>
        -
		</td>
      <td>
        -
		</td>
	</tr>
    <tr align="center" valign="center">
      <td>
        <b>W-G EKF</b>
      </td>
      <td>
        -
		</td>
      <td>
        18.76
		</td>
      <td>
        12.29
		</td>
      <td>
        8.94
		</td>
      <td>
        13.33
		</td>
      <td>
        -
		</td>
      <td>
        -
		</td>
	</tr>
    <tr align="center" valign="center">
      <td>
        <b>W-G KNetArch1</b>
      </td>
      <td>
        1.3 M
		</td>
      <td>
        15.520
		</td>
      <td>
        7.806
		</td>
      <td>
        7.087
		</td>
      <td>
        10.137
		</td>
      <td>
        10.961
		</td>
      <td>
        <a href="./configs/nclt_fusion/knet_arch1.py">config</a>
		</td>
		</tr>
    <tr align="center" valign="center">
      <td>
        <b>W-G KNetArch2</b>
      </td>
      <td>
        107 K
		</td>
      <td>
        14.899
		</td>
      <td>
        8.916
		</td>
      <td>
        8.490
		</td>
      <td>
        10.768
		</td>
      <td>
        11.256
		</td>
      <td>
        <a href="./configs/nclt_fusion/knet_arch2.py">config</a>
		</td>
		</tr>
    <tr align="center" valign="center">
      <td>
        <b>W-G SKNet</b>
      </td>
      <td>
        463 K
		</td>
      <td>
        16.105
		</td>
      <td>
        10.037
		</td>
      <td>
        6.532
		</td>
      <td>
        10.891
		</td>
      <td>
        11.762
		</td>
      <td>
        <a href="./configs/nclt_fusion/sknet.py">config</a>
		</td>
		</tr>
    <tr align="center" valign="center">
      <td>
        <b>W-G SIKNet</b>
      </td>
      <td>
        453 K
		</td>
      <td>
        14.434
		</td>
      <td>
        7.687
		</td>
      <td>
        5.999
		</td>
      <td>
        9.374
		</td>
      <td>
        10.399
		</td>
      <td>
        <a href="./configs/nclt_fusion/siknet.py">config</a>
		</td>
		</tr>
  </tbody>
</table>

## Getting Started

### Installation

Please refer to [Installation](./docs/en/Installation.md).

### Supported Datasets

Please refer to [Datasets](./docs/en/Datasets.md).

### Training

Please refer to [Training](./docs/en/Training.md).

## Citation

If you find this repo useful, please cite our papers.
<a id="anchor1"></a>

```bibtex
@ARTICLE{10605082,
  author={Song, Jian and Mei, Wei and Xu, Yunfeng and Fu, Qiang and Bu, Lina},
  journal={IEEE Signal Processing Letters},
  title={Practical Implementation of KalmanNet for Accurate Data Fusion in Integrated Navigation},
  year={2024},
  volume={31},
  number={},
  pages={1890-1894},
  keywords={Training;Sensor fusion;Global Positioning System;Navigation;Vectors;Kalman filters;Wheels;Integrated navigation and localization;Kalman filter;recurrent neural networks;sensor fusion},
  doi={10.1109/LSP.2024.3431443}}
```

Others

## Acknowledgement

The structure of this repository and much of the code is thanks to the authors of the following repositories.

- [filterpy](https://github.com/rlabbe/filterpy): A really great (I think the best) python based filter repository. Also has a filter [teaching repository](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) with it.
- [torchfilter](https://github.com/stanford-iprl-lab/torchfilter): Is a library for discrete-time Bayesian filtering in PyTorch. By writing filters as standard PyTorch modules.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SongJgit/filternet&type=Date)](https://www.star-history.com/#SongJgit/filternet&Date)
