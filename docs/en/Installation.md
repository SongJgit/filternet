# Installation

## Note

1. We use lightning to simplify the writing of project code, so we need to be careful about the correspondence between the version of lightning and pytorch when installing, the version of lightning that matches pytorch can be found \[here\](https://lightning.ai/docs/pytorch/latest/versioning.html#compatibility-matrix).

2. lighting.pytorch comes after pytorch_lightning = 1.8, so this project requires at least lighting >= 1.8, pytorch >= 1.10.

## Started

1. Create environment & activate environment

- Create environment

  Best practice

  ```bash
  conda create -n filternet python==3.10
  ```

  or at least

  ```bash
  conda create -n filternet python==3.8
  ```

- Activate environment

  ```bash
  conda activate filternet
  ```

2. Install this repo
   ```bash
   git clone git@github.com:SongJgit/filternet.git
   cd filternet
   pip install -e .
   ```
