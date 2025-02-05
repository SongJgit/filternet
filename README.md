# FilterNet - Data-Driven Filtering in Python(We will upload the full code once the paper has been accepted)

## Introduction

This library provides Data-Driven Kalman filtering and various related optimal and non-optimal filtering software written in Python.
It contains Kalman filters, Extended Kalman filters, KalmanNet, Split-KalmanNet and Ours Semantic-Independent KalmanNet.
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

## Getting Started

### Installation

Please refer to [Installation](./docs/en/Installation.md)
