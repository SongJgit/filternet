import torch
import torch.nn as nn

from filternet.registry import MEASUREMENTMODEL


@MEASUREMENTMODEL.register_module()
class LinearKalmanFilterMeasurementModel(nn.Module):

    def __init__(self,
                 dim_state: int,
                 dim_obs: int,
                 constant_noise: bool = False,
                 device: str = 'cuda'):
        super().__init__()

        self.dim_state = dim_state
        self.dim_obs = dim_obs
        self.initialized_matrices = False
        self.constant_noise = constant_noise
        self.device = device

    def initialize_matrices(self, params) -> None:
        self.params = params
        R = self.params.R
        if R.ndim > 3:
            assert self.constant_noise is False, f'if not {self.constant_noise=} R must [..., T]'
            self._step = 0
        else:
            assert self.constant_noise, f'if {self.constant_noise=} R must \
            [{self.dim_obs, self.dim_obs}]'

        self.R: torch.Tensor = R.to(self.device)
        self.initialized_matrices = True

    def forward(self, *, state: torch.Tensor):
        assert self.initialized_matrices
        N, dim_state, _ = state.shape
        assert self.dim_state == dim_state

        obs_prior, jac = self.params.get_obs_jac(state)
        if not self.constant_noise:
            T = self.R.shape[-1]
            assert self._step <= T, f'if not constant_nosie, {self._step=} must be smaller than T'
            R = self.R[..., self._step]
            assert R.shape == (N, self.dim_obs, self.dim_obs)
            self._step += 1
        else:
            R = self.R[None, ...].expand(N, self.dim_obs, self.dim_obs).clone()
        return obs_prior, jac[..., -1], R.to(self.device)


@MEASUREMENTMODEL.register_module()
class EKFMeasurementModel(nn.Module):

    def __init__(self,
                 dim_state: int,
                 dim_obs: int,
                 constant_noise: bool = False,
                 device: str = 'cuda'):
        super().__init__()

        self.dim_state = dim_state
        self.dim_obs = dim_obs
        self.initialized_matrices = False
        self.constant_noise = constant_noise
        self.device = device

    def initialize_matrices(self, params) -> None:
        self.params = params
        # assert H.shape == (self.dim_obs, self.dim_state)
        # self.H: torch.Tensor = H[None, ...].to(self.device)
        R = self.params.R
        if R.ndim > 3:
            assert self.constant_noise is False, f'if not {self.constant_noise=} R must [..., T]'
            self._step = 0
        else:
            assert self.constant_noise, f'if {self.constant_noise=} R must \
            [{self.dim_obs, self.dim_obs}]'

        self.R: torch.Tensor = R.to(self.device)

        self.initialized_matrices = True

    def forward(self, *, state: torch.Tensor):
        assert self.initialized_matrices
        N, dim_state, _ = state.shape
        assert self.dim_state == dim_state

        obs_prior, jac = self.params.get_obs_jac(state)
        R = self.R[None, ...].expand(N, self.dim_obs,
                                     self.dim_obs).clone().to(self.device)
        return obs_prior, jac[..., -1], R
