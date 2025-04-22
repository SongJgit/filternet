<p align="center">
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>

</p>

# FilterNet - Learning-Aided Filtering in Python

## Notice

We will upload all the code once the paper submitted to "Information Fusion" has been accepted.

You can view papers related to Learning Inspired Filtering through the following [links](https://github.com/SongJgit/awesome-learning-aided-filter-papers).

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
          <li><a href="">SmoothL1oss</a></li>
          <li><a href="https://ieeexplore.ieee.org/document/10485649/">DanseLoss</a></li>
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

## Supervised Learning or Unsupervised Learning?

| **Methods** | **Supervised Learning** | **Unsupervised Learning** |
|:-------------------:|:-----------------------:|:-------------------------:|
| **KalmanNet** | ✔ | ✔ |
| **Split-KalmanNet** | ✔ | ✔ |
| **Danse** | ✔ | ✔ |

## Getting Started

### Installation

Please refer to [Installation](./docs/en/Installation.md).

### Supported Datasets

Please refer to [Datasets](./docs/en/Datasets.md).

### Training

Please refer to [Training](./docs/en/Training.md).

## Citation

If you find this repo useful, please cite our papers.

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
