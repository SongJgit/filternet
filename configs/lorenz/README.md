## Lorenz Attractor

This dataset comes from the paper: [KalmanNet: neural network aided kalman filtering for partially known dynamics](https://ieeexplore.ieee.org/document/9733186/). [Citation Key](#anchor1)

The code is modified from the paper : [DANSE: Data-Driven Non-Linear State Estimation of Model-Free Process in Unsupervised Learning Setup](https://ieeexplore.ieee.org/document/10485649/). [Citation Key](#anchor2).

To facilitate observation, we did not generate data with a `dB` setting, but based directly on the noise variance. The parameters setting are based on the paper: [Physics-informed data-driven autoregressive nonlinear filter](https://ieeexplore.ieee.org/document/10884033/). [Citation Key](#anchor3).

### Preparation of data

Recommended to check the documentation here: [ROOT_DIR/datasets_tools/lorenz_datasets/README.md](../../datasets_tools/lorenz_datasets/README.md)

### Training & Validation & Testing

```bash
python tools/train.py --cfg ./configs/lorenz/knet_arch2.py
```

In the end of the training, you can view the results in CLI

### Citation

<a id="anchor1"></a>

```bibtex
@article{2022revachkalmannetneuralnetwork,
  title = {KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics},
  shorttitle = {KalmanNet},
  author = {Revach, Guy and Shlezinger, Nir and Ni, Xiaoyong and Escoriza, Adria Lopez and Van Sloun, Ruud J. G. and Eldar, Yonina C.},
  year = {2022},
  journal = {IEEE Trans. Signal Process.},
  volume = {70},
  pages = {1532--1547},
  issn = {1053-587X, 1941-0476},
  doi = {10.1109/TSP.2022.3158588},
  urldate = {2023-10-11},
  langid = {english},
  lccn = {2},
  keywords = {ObsCite}
}
```

<a id="anchor2"></a>

```bibtex
@article{2024ghoshdansedatadrivennonlinear,
  title = {DANSE: Data-Driven Non-Linear State Estimation of Model-Free Process in Unsupervised Learning Setup},
  author = {Ghosh, Anubhab and Honor{\'e}, Antoine and Chatterjee, Saikat},
  year = {2024},
  journal = {IEEE Trans. Signal Process.},
  volume = {72},
  pages = {1824--1838},
  issn = {1053-587X, 1941-0476},
  doi = {10.1109/TSP.2024.3383277},
  copyright = {https://github.com/saikatchatt/danse-jrnl},
  langid = {english},
  lccn = {2},
  keywords = {Bayes methods,Bayesian state estimation,Computational modeling,forecasting,neural networks,Noise measurement,recurrent neural networks,State estimation,Supervised learning,Training,unsupervised learning,Unsupervised learning}
}
```

<a id="anchor3"></a>

```bibtex
@article{2025liuphysicsinformeddatadrivenautoregressive,
  title = {Physics-Informed Data-Driven Autoregressive Nonlinear Filter},
  author = {Liu, Hanyu and Sun, Xiucong and Chen, Yuran and Wang, Xinlong},
  year = {2025},
  journal = {IEEE Signal Processing Letters},
  volume = {32},
  pages = {846--850},
  issn = {1070-9908, 1558-2361},
  doi = {10.1109/LSP.2025.3541537},
  urldate = {2025-03-12},
  copyright = {https://ieeexplore.ieee.org/Xplorehelp/downloads/license-information/IEEE.html},
  langid = {english},
  lccn = {2}
}
```
