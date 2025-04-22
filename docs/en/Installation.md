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

2. Install Pytorch and Lightning.

- Best practice

  pytorch

  ```bash
  conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
  ```

  lightning

  ```bash
  pip install lightning==2.3.0
  ```

- or at least

  pytorch

  ```bash
  pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
  ```

  lightning

  ```bash
  pip install lightning==1.9.0
  ```

3. Install this repo.

   ```bash
   git clone git@github.com:SongJgit/filternet.git
   ```

   and

   ```bash
   cd filternet
   ```

   Install this repo with pip.

   ```bash
   pip install -e .
   ```
