from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from filternet.registry import FILTER, PARAMS
from filternet.utils.logger import global_logger as logger

# torch.set_default_dtype(torch.float64)


@FILTER.register_module()
class KalmanFilter(nn.Module):

    def __init__(self, params: Dict[str, Any] | Any, device: str = 'cuda'):
        super().__init__()
        self.device = device
        # init params
        if isinstance(params, dict):
            self.params = PARAMS.build(params)
        else:
            self.params = params

        self.dim_state = self.params.dim_state
        self.dim_obs = self.params.dim_obs

        # self.state_post: torch.Tensor
        # self.state_prior: torch.Tensor
        # self.P_post: torch.Tensor
        # self.P_prior: torch.Tensor
        # self.state: torch.Tensor
        # self.P: torch.Tensor

        self._initialized = False

    def init_beliefs(self, state: torch.Tensor, covariance: torch.Tensor | None = None) -> None:
        """_summary_

        Args:
            state (torch.Tensor): [N, num_states]
            covariance (torch.Tensor):[N,.num_states, num_states]
        """
        assert state.ndim == 3, f'state must be in batch form [B, dim_state, 1], but got {state.ndim} dimensions'

        self.device = state.device
        self.batch_size = state.shape[0]
        assert state.shape == (self.batch_size, self.dim_state, 1)

        if covariance is None:
            covariance = self.params.P
        elif covariance.ndim == 3:
            # must be [B, dim_state, dum_state]
            self.params.P = covariance
        else:
            # must be [B, dim_state, dum_state]
            covariance = covariance.unsqueeze(0)
            self.params.P = covariance

        assert covariance.shape[-2:] == (self.dim_state, self.dim_state)

        self.params.to(self.device)
        self.state_post = state.to(self.device)
        self.P_post = covariance.to(self.device)
        self._initialized = True

    def forward(self, obs: torch.Tensor):
        assert self._initialized, 'Kalman filter not initialized!'

        self.predict_step()
        self.update_step(obs)

        return self.state_post

    def predict_step(self):
        prev_mean = self.state_post
        prev_covariance = self.P_post
        N, dim_state, _ = prev_mean.shape

        pred_mean, F_matrix = self.params.get_pred_jac(prev_mean)
        F_matrix = F_matrix.squeeze(-1)
        pred_covariance = F_matrix @ prev_covariance @ F_matrix.transpose(-1, -2) + self.Q.to(self.device)

        self.state_post = pred_mean
        self.P_post = pred_covariance
        return self.state_post, self.P_post

    def update_step(self, obs: torch.Tensor | None):
        """_summary_

        Args:
            obs (torch.Tensor | None): [B, dim_obs]

        Returns:
            _type_: _description_
        """

        pred_mean = self.state_post
        pred_covariance = self.P_post

        obs_prior, H_matrix = self.params.get_obs_jac(pred_mean)
        self.obs_prior = obs_prior
        H_matrix = H_matrix.squeeze(-1)

        self.Innov = obs - obs_prior
        PHT = pred_covariance @ H_matrix.transpose(-1, -2)

        S = H_matrix @ PHT + self.R.to(self.device)
        S = 0.5 * (S + S.transpose(-1, -2))  # symmetrize
        self.Sk = S
        # for stable, use pininverse.
        kalman_gain = PHT @ torch.pinverse(S)

        corrected_state = torch.baddbmm(pred_mean, kalman_gain, self.Innov)

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation P = (I-KH)P usually seen in the literature.

        identity = torch.eye(self.dim_state, device=self.device)
        I_KH = identity - kalman_gain @ H_matrix

        corrected_covariance = I_KH @ pred_covariance @ I_KH.transpose(-1, -2) + kalman_gain @ self.R.to(
            self.device) @ kalman_gain.transpose(-1, -2)

        self.state_post = corrected_state
        self.P_post = corrected_covariance
        return self.state_post, self.P_post

    def forward_loop(self, obs: torch.Tensor) -> torch.Tensor:
        """

        Args:
            obs (torch.Tensor): [batch_size, obs, T]

        Returns:
            torch.Tensor: _description_
        """
        obs = obs.to(self.device)
        batch_size, dim_obs, T = obs.shape
        assert dim_obs == self.dim_obs, f'obs must have dimension {self.dim_obs=}, but got {dim_obs=}'
        assert batch_size == self.batch_size, f'obs must have batch size {self.batch_size}, but got {batch_size}'

        self.state_post_seq = torch.zeros([self.batch_size, self.dim_state, T], device=self.device)
        self.P_post_seq = torch.zeros([self.batch_size, self.dim_state, self.dim_state, T], device=self.device)
        self.S_seq = torch.zeros([self.batch_size, self.dim_obs, self.dim_obs, T], device=self.device)

        for t in range(T):
            current_prediction = self(obs=obs[..., [t]])
            self.state_post_seq[..., [t]] = current_prediction
            self.P_post_seq[..., t] = self.P_post
            self.S_seq[..., t] = self.Sk

        return self.state_post_seq

    @property
    def F(self):
        return self.params.F

    @F.setter
    def F(self, F: torch.Tensor):
        if F.ndim == 2:
            F = F.unsqueeze(0)
        self.params.F = F.to(self.device)

    @property
    def H(self):
        return self.params.H

    @H.setter
    def H(self, H: torch.Tensor):
        if H.ndim == 2:
            H = H.unsqueeze(0)
        self.params.H = H.to(self.device)

    @property
    def Q(self):
        return self.params.Q

    @Q.setter
    def Q(self, Q: torch.Tensor):
        if Q.ndim == 2:
            Q = Q.unsqueeze(0)
        self.params.Q = Q.to(self.device)

    @property
    def R(self):
        return self.params.R

    @R.setter
    def R(self, R: torch.Tensor):
        if R.ndim == 2:
            R = R.unsqueeze(0)
        self.params.R = R.to(self.device)

    @property
    def P(self):
        return self.params.P

    @P.setter
    def P(self, P: torch.Tensor):
        if self._initialized:
            logger.error('Cannot set P directly after initialization.')

        else:
            if P.ndim == 2:
                P = P.unsqueeze(0)
        self.params.P = P.to(self.device)

    def get_pred_jac(self, state: torch.Tensor):
        return self.params.get_pred_jac(state)

    def get_obs_jac(self, obs: torch.Tensor):
        return self.params.get_obs_jac(obs)

    def __repr__(self):
        return f'KalmanFilter({self.params})'


if __name__ == '__main__':
    from filternet.params import CVParams

    length = 10
    dim_state = 4
    obs_axis = [0, 2]
    batch_size = 2
    init_state = torch.randn(batch_size, dim_state, 1)  # [B, dim_state, 1]
    obs = torch.randn(batch_size, len(obs_axis), length)  # [B, dim_obs, T]
    P = torch.eye(dim_state)

    params = dict(type=CVParams, dim_state=dim_state, obs_ind=obs_axis, noise_q2=1, noise_r2=1)
    cv_filter = dict(type=KalmanFilter, params=params, device='cpu')
    filter = FILTER.build(cv_filter)
    filter.init_beliefs(init_state)
    est = filter.forward_loop(obs)  # [B, dim_state, T]
    assert est.shape == (batch_size, dim_state, length)
