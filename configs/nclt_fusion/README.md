## NCLT Fusion

This dataset comes from the paper: [University of michigan north campus long-term vision and lidar dataset](http://journals.sagepub.com/doi/10.1177/0278364915614638). [Citation Key](#anchor1)

The Model-based EKF implementation and methods of data pre-processing comes from this repository [mte546-project](https://github.com/AbhinavA10/mte546-project). [Citation Key](#anchor2)

The Learning-Aided Kalman filtering implementation comes from the paper [Practical implementation of KalmanNet for accurate data fusion in integrated navigation](https://ieeexplore.ieee.org/document/10605082). [Citation Key](#anchor3)

### Preparation of data

Recommended to check the documentation here: [ROOT_DIR/datasets_tools/nclt_fusion/README.md](../../datasets_tools/nclt_fusion/README.md)

### Training & Validation & Testing

```bash
python tools/train.py --cfg ./configs/nclt_fusion/knet_arch2.py
```

In the end of the training, you can view the results in CLI

```bash
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test/MSEdB         │     21.02839469909668     │
│         test/RMSE         │    11.256922721862793     │
│      test_loss/loss       │    126.71831512451172     │
│      test_loss/main       │    126.71831512451172     │
└───────────────────────────┴───────────────────────────┘
```

And, you can find the trained model in `ROOT_DIR/runs/nclt_fusion/KNetArch2_unsupFalse_v0/checkpoints`

### Plot

View the `ROOT_DIR/notebook/nclt_fusion_predict.ipynb`

### Citation

If you find this repo useful, please cite these papers.

<a id="anchor1"></a>

```bibtex
@article{2016carlevaris-biancouniversitymichigannorth,
  title = {University of Michigan North Campus Long-Term Vision and Lidar Dataset},
  author = {{Carlevaris-Bianco}, Nicholas and Ushani, Arash K and Eustice, Ryan M},
  year = {2016},
  month = aug,
  journal = {International Journal of Robotics Research},
  volume = {35},
  number = {9},
  pages = {1023--1035},
  issn = {0278-3649, 1741-3176},
  doi = {10.1177/0278364915614638},
  urldate = {2023-10-26},
  langid = {english}
}
```

<a id="anchor2"></a>

```bibtex
@misc{2023agraharilocalizationmobilerobot,
  title = {Localization of Mobile Robot Using Classical Bayesian Filtering and Sensor Data},
  author = {Agrahari, Abhinav and Barlow, Andrew and Naik, Sameeksha and Skarica, Stephanie},
  year = {2023},
  howpublished = {2023. [Online]. Available: https://github.com/AbhinavA10/mte546-project},
  langid = {english},
  keywords = {No DOI found}
}
```

<a id="anchor3"></a>

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
