from __future__ import annotations

from copy import deepcopy

from typing import Any

import torch
import torch.nn as nn

from filternet.models.filter.kalman_filter import KalmanFilter
from filternet.registry import FILTER


@FILTER.register_module()
class MOTKalmanFilter(KalmanFilter):
    """Examples:
            from filternet.dataset.track_dataset import RadarTrackDataset
            dataset=dict(type = RadarTrackDataset,
                        mode = 'val',
                        data_root = './data/radar_ca_dims9.pt',
                        seq_len = 100)
            params=dict(
                    type = DataParams,
                    dataset =dataset)

            measurement_model =dict(
                type = LinearKalmanFilterMeasurementModel,
                dim_state = 9,
                obs =3,
                device= 'cuda'
            )
            dynamics_model =dict(
                type = LinearDynamicsModel,
                dim_state = 9,
                device= 'cuda')
            filter = dict(
                type = LinearKalmanFilter,
                dynamics_model =dynamics_model,
                measurement_model =measurement_model,
                params = params,
                device= 'cuda'
            )
            cfg = Config(filter)
            kf = FILTER.build(cfg)
        """

    def init_beliefs(self, state: torch.Tensor, covariance: torch.Tensor | None = None) -> None:
        """_summary_

        Args:
            state (torch.Tensor): [N, num_states]
            covariance (torch.Tensor):[N,.num_states, num_states]
        """
        N = state.shape[0]
        self.params = deepcopy(self._params)
        self.dynamics_model.initialize_matrices(self.params)
        self.measurement_model.initialize_matrices(self.params)

        state, init_covariance = self.params.initiate(state)

        if covariance is None:
            covariance = init_covariance
        assert state.shape == (N, self.dim_state, 1), f'{self.dim_state=}, {state.shape=}'
        assert covariance.shape == (N, self.dim_state, self.dim_state)

        self.state_post = state.to(self.device)
        self.P_post = covariance.to(self.device)

        self._initialized = True

    def update_step(self, obs: torch.Tensor | None):

        # for nan obs
        nan_mask = obs.isnan().any(dim=1).flatten()  # if nan, is true [bs]
        new_obs = torch.zeros_like(obs)
        new_obs[~nan_mask] = obs[~nan_mask].clone()
        obs = new_obs

        pred_mean = self.state_post
        pred_covariance = self.P_post
        N, dim_state, _ = pred_mean.shape
        N, dim_obs, _ = obs.shape
        # BUG: In this, only support Linear.
        obs_prior, H_matrix, R = self.measurement_model(state=pred_mean)

        assert H_matrix.shape == (N, self.dim_obs, self.dim_state)
        assert obs_prior.shape == (N, self.dim_obs, 1)
        assert R.shape == (N, self.dim_obs, self.dim_obs), f'{R.shape=}'
        # R = noise_r2_matrix * self.params.noise_r2
        # compute innov, gain

        self.Innov = obs - obs_prior
        self.Sk = H_matrix @ pred_covariance @ H_matrix.transpose(-1, -2) + R

        # [bs, dim_state, dim_state] @ [bs, dim_state,dim_obs]@[bs, dim_obs,dim_obs]
        # Method1
        # gain = pred_covariance @ H_matrix.transpose(-1, -2) @ torch.linalg.inv(
        #     self.Sk)
        # Method2
        gain = pred_covariance @ H_matrix.transpose(-1, -2) @ torch.linalg.pinv(self.Sk)
        # Method3, from deepsort
        # L = torch.linalg.cholesky_ex(self.Sk)
        # gain = torch.cholesky_solve((pred_covariance @ H_matrix.transpose(-1, -2)).transpose(-1,-2), \
        # L).transpose(-1,-2)

        assert gain.shape == (N, self.dim_state, self.dim_obs)
        # Get mu_{t+1|t+1}, Sigma_{t+1|t+1}

        correct_state = torch.baddbmm(pred_mean, gain, self.Innov)
        assert correct_state.shape == (N, self.dim_state, 1)

        # correct_covariance = pred_covariance - torch.bmm(
        #     gain, torch.bmm(self.Sk, gain.transpose(-1, -2)))
        # Ensure positive definiteness and symmetry of the covariance matrix
        identity = torch.eye(self.dim_state, device=gain.device)
        correct_covariance = (identity - gain @ H_matrix) @ pred_covariance @ (identity + gain @ H_matrix).transpose(
            -1, -2) - gain @ R @ gain.transpose(-1, -2)

        assert correct_covariance.shape == (N, self.dim_state, self.dim_state)

        self.state_post[~nan_mask] = correct_state[~nan_mask].clone()
        self.P_post[~nan_mask] = correct_covariance[~nan_mask].clone()
