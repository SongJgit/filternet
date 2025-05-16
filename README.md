<p align="center">
<a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" alt="pre-commit" style="max-width:100%;"></a>
<a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" style="max-width:100%;"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>

</p>

# FilterNet - Learning-Aided Filtering in Python

## Notice

We will upload all the code once the paper submitted to "Information Fusion" has been accepted.

You can view papers related to Learning-Aided Filtering through the following [links](https://github.com/SongJgit/awesome-learning-aided-filter-papers).

## 🥳 What's New

- Feb. 2025: 🌟🌟🌟 First commit. Added support for model-based Kalman filter, Extended Kalman filter, Interacting Multiple model, and learning-aided Kalman filtering KalmanNet, Split-KalmanNet.

## Introduction

This library provides Learning-Aided/Data-Driven Kalman filtering and related optimal and non-optimal filtering software in Python.
It contains Kalman filters, Extended Kalman filters, KalmanNet, Split-KalmanNet, and Ours Semantic-Independent KalmanNet(submitted to Information Fusion, waiting review).
This library is implemented with **[Pytorch-Lightning](https://lightning.ai/docs/pytorch/stable/)**, **[MMEngine](https://github.com/open-mmlab/mmengine)**, and **[WandB](https://wandb.ai/site)**.

## Highlights

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
- system state dimension: $m = 3$, observation dimension: $n = 3$.

Note: In order to compare with other models, DANSE is trained using a supervised method from the source code.

|    Methods    | Params | RMSE@1  | RMSE@10 | RMSE@100 | RMSE@1000 |
| :-----------: | :----: | :-----: | :-----: | :------: | :-------: |
| **Obs Error** |  None  |  2.31   |  3.78   |  10.26   |   31.56   |
|   **KNet**    | 366 K  | 0.60431 | 1.18635 |  2.8958  |    Nan    |
|   **SKNet**   | 149 K  | 0.39873 | 0.91222 | 2.01605  |  5.37797  |
|   **DANSE**   | 4.3 K  | 0.59011 | 1.20016 | 3.11831  |  7.99238  |
|  **SIKNet**   | 140 K  | 0.49095 | 0.82325 | 2.03247  |  5.28136  |

### NCLT Sensor Fusion

- This implementation is sources from [Practical Implementation of KalmanNet for Accurate Data Fusion in Integrated Navigation](https://ieeexplore.ieee.org/document/10605082). If it's helpful, please cite the [paper](#anchor1).

- [More Details](configs/nclt_fusion/README.md).

- [WandB Logger](https://wandb.ai/songj/NCLT_Fusion_Benchmark?nw=nwusersongj).

- $\mathrm{AvgRMSE@All} = Average(\mathrm{RMSE@Traj1}, \mathrm{RMSE@Traj2}, \cdots, \mathrm{RMSE@TrajN})$.

- $\mathrm{RMSE@All} = RMSE(\mathrm{Traj1}, \mathrm{Traj2}, \cdots, \mathrm{TrajN})$.

<table style="border-collapse: collapse; border: none; border-spacing: 0px;" align="center">
	<tr>
		<td colspan="2" style="border-bottom: 1px solid rgb(0, 0, 0); border-top: 2px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>Metric</b>
		</td>
		<td style="border-top: 2px solid rgb(0, 0, 0); border-bottom: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>RMSE@2012-11-04</b>
		</td>
		<td style="border-top: 2px solid rgb(0, 0, 0); border-bottom: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>RMSE@2012-11-16</b>
		</td>
		<td style="border-top: 2px solid rgb(0, 0, 0); border-bottom: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>RMSE@2013-04-05</b>
		</td>
		<td style="border-top: 2px solid rgb(0, 0, 0); border-bottom: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>AvgRMSE@All</b>
		</td>
		<td style="border-top: 2px solid rgb(0, 0, 0); border-bottom: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b><b>RMSE@All</b></b>
		</td>
		<td style="border-top: 2px solid rgb(0, 0, 0); border-bottom: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>Config</b>
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>Methods</b>
		</td>
		<td style="padding-right: 3pt; padding-left: 3pt; background-image: linear-gradient(to right top, rgba(255, 255, 255, 0) 0%, rgba(255, 255, 255, 0) 49.9%, rgb(0, 0, 0) 50%, rgb(0, 0, 0) 51%, rgba(255, 255, 255, 0) 51.1%, rgba(255, 255, 255, 0) 100%);">
			<div style="padding-left: 50px;word-break: keep-all;white-space: nowrap;">
				<b>Traj.Len[Sec]</b>
			</div>
			<div style="padding-right:50px;word-break: keep-all;white-space: nowrap;">
				<b>Params</b>
			</div>
		</td>
		<td style="border-bottom: 1px solid rgb(0, 0, 0); border-top: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			4834
		</td>
		<td style="border-bottom: 1px solid rgb(0, 0, 0); border-top: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			4917
		</td>
		<td style="border-bottom: 1px solid rgb(0, 0, 0); border-top: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			4182
		</td>
		<td style="border-bottom: 1px solid rgb(0, 0, 0); border-top: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			-
		</td>
		<td style="border-bottom: 1px solid rgb(0, 0, 0); border-top: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			-
		</td>
		<td style="border-bottom: 1px solid rgb(0, 0, 0); border-top: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			-
		</td>
	</tr>
	<tr>
		<td style="border-top: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>GPS-Only</b>
		</td>
		<td style="border-top: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			-
		</td>
		<td style="border-top: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			46.141
		</td>
		<td style="border-top: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			19.500
		</td>
		<td style="border-top: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			34.538
		</td>
		<td style="border-top: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			33.393
		</td>
		<td style="border-top: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			35.084
		</td>
		<td style="border-top: 1px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			-
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>Wheels-Only EKF</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			-
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		117.76
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		81.61
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		83.99
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		94.46
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		-
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		-
		</td>
	</tr>
		<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>Wheels-GPS EKF</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			-
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		18.76
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		12.29
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		8.94
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		13.33
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		-
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		-
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>Wheels-GPS KNetArch1</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			1.3 M
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		15.520
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		7.806
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		7.087
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		10.137
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		10.961
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
	    <a href="./configs/nclt_fusion/knet_arch1.py">config</a>
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>Wheels-GPS KNetArch2</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			107 K
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		14.899
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		8.916
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		8.490
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		10.768
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		11.256
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		<a href="./configs/nclt_fusion/knet_arch2.py">config</a>
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>Wheels-GPS SKNet</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			463 K
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		16.105
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		10.037
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		6.532
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		10.891
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		11.762
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		<a href="./configs/nclt_fusion/sknet.py">config</a>
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 2px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>Wheels-GPS SIKNet</b>
		</td>
		<td style="border-bottom: 2px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			453 K
		</td>
		<td style="border-bottom: 2px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">14.434
		</td>
		<td style="border-bottom: 2px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">7.687
		</td>
		<td style="border-bottom: 2px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">5.999
		</td>
		<td style="border-bottom: 2px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">9.374
		</td>
		<td style="border-bottom: 2px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
		10.399
		</td>
		<td style="border-bottom: 2px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
		<a href="./configs/nclt_fusion/siknet.py">config</a>
		</td>
	</tr>
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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SongJgit/filternet&type=Date)](https://www.star-history.com/#SongJgit/filternet&Date)
