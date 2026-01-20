from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import torch
import numpy as np
import torch.nn as nn
from jax.scipy.stats import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal
from filternet.utils import global_logger as logger
from filternet.registry import FILTER, MODELTRANS


@FILTER.register_module()
class IMMFilter(nn.Module):

    def __init__(self,
                 filters: List[Dict[str, Any]] | List[Any],
                 trans_prob: List[Any],
                 model_prob: List[Any],
                 dim_state: int,
                 dim_obs: int,
                 model_trans: Dict[str, Any] | List[Any] = None,
                 device='cuda',
                 trans_init: bool = True):
        super().__init__()
        """By default the last model has the largest dimension
        """
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        self.device = device
        self.trans_init = trans_init
        if isinstance(filters[0], dict):
            self.filters = [FILTER.build(filter) for filter in filters]
        else:
            self.filters = filters
        self.num_filters = len(self.filters)

        if isinstance(model_trans, dict):
            self.model_trans = MODELTRANS.build(model_trans)
        else:
            self.model_trans = model_trans

        if self.model_trans is None or (hasattr(self.model_trans, '__class__')
                                        and self.model_trans.__class__.__name__ == 'SelfTrans'):
            logger.info('Detected SelfTrans or null model_trans: using fully vectorized fast IMM path.')
            self.predict_step = self._fast_predict_step
            self._compute_state_estimate = self._fast_compute_state_estimate
            self._fast_path = True
        else:
            logger.info('Using generic IMM path with arbitrary model transitions.')
            self.predict_step = self._generic_predict_step
            self._compute_state_estimate = self._generic_compute_state_estimate
            self.model_trans.to(self.device)
            assert len(self.model_trans) == self.num_filters
            self._fast_path = False

        # for convenience, using column vector.
        self._init_model_prob = torch.tensor(model_prob).reshape(-1, 1)[None, ...].to(self.device)  # [batch,...]
        self._init_trans_prob = torch.tensor(trans_prob)[None, ...].to(self.device)
        #
        assert self._init_trans_prob.shape == (1, self.num_filters, self.num_filters)
        assert self._init_model_prob.shape == (1, self.num_filters, 1)

        self._init_Omega = torch.zeros_like(self._init_trans_prob).to(self.device)
        self._initialized = False

    def init_beliefs(self, state: torch.Tensor, covariance: torch.Tensor | None = None) -> None:
        self.batch_size = state.shape[0]
        assert state.shape == (self.batch_size, self.dim_state, 1)

        if covariance is None:
            covariance = torch.eye(self.dim_state)[None, ...].expand(self.batch_size, self.dim_state,
                                                                     self.dim_state).clone()

        assert covariance.shape == (self.batch_size, self.dim_state, self.dim_state)
        state = state.to(self.device)
        covariance = covariance.to(self.device)
        self.model_prob = self._init_model_prob.expand(self.batch_size, self.num_filters, 1).clone()
        self.trans_prob = self._init_trans_prob.expand(self.batch_size, self.num_filters, self.num_filters).clone()
        self.Omega = self._init_Omega.expand_as(self.trans_prob).clone()

        # Use the last filter to initialize the follow-up model.
        if self._fast_path:
            for i, filter in enumerate(self.filters):
                filter.init_beliefs(state, covariance)
        else:
            for i, filter in enumerate(self.filters):
                transed_state = self.model_trans[i][-1][None, ...] @ state
                transed_covariance = self.model_trans[i][-1][None, ...] @ covariance @ self.model_trans[i][-1][
                    None, ...].transpose(-1, -2)
                filter.init_beliefs(transed_state, transed_covariance)

        self._compute_mixing_probabilities()

        # compute mixed IMM state and covariance and save posterior estimate
        self._compute_state_estimate()

        self._initialized = True

    def _fast_predict_step(self):
        """Fully vectorised only model_trans is SelfTrans, the fastest method
        available, but with one fundamental requirement:

        foremost among them is the requirement that all of the filters in the bank have the same dimensional design.
        """
        assert self._initialized

        # Stack states and covariances
        X = torch.stack([f.state_post for f in self.filters], dim=1)  # (B, M, d, 1)
        P = torch.stack([f.P_post for f in self.filters], dim=1)  # (B, M, d, d)

        # Weight: W[b, i, j] = Omega[b, j, i]
        W = self.Omega.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)  # (B, M, M, 1, 1)

        print(W[0, :, :, 0, 0])
        # Mixed mean: μ_i = Σ_j Omega[j,i] * x_j
        # X[:, None, :, :, :] → (B, 1, M, d, 1) → broadcast over i
        state_mix = (X[:, None, :, :, :] * W).sum(dim=2)  # (B, M, d, 1)

        # Mixed covariance: P_i = Σ_j Omega[j,i] * [ P_j + (x_j - μ_i)(x_j - μ_i)^T ]
        diff = X[:, None, :, :, :] - state_mix[:, :, None, :, :]  # (B, M, M, d, 1)
        diff_outer = diff @ diff.transpose(-1, -2)  # (B, M, M, d, d)

        P_expanded = P[:, None, :, :, :]  # (B, 1, M, d, d)
        P_total = P_expanded + diff_outer  # (B, M, M, d, d)

        P_mix = (P_total * W).sum(dim=2)  # (B, M, d, d)

        # step2: filt
        for j in range(self.num_filters):
            self.filters[j].state_post = state_mix[:, j, ...].clone()
            self.filters[j].P_post = P_mix[:, j, ...].clone()
            self.filters[j].predict_step()

        self._compute_state_estimate()

    def _generic_predict_step(self):
        assert self._initialized
        # N, dim_obs, _ = x.shape

        # step1: mix probability
        # C^j=\sum_{i=1}^np_{ij}U_{k-1}^i

        # state_mix = torch.zeros(N, self.dim_state, self.num_filters)
        state_mix = [torch.zeros_like(filter.state_post, device=filter.state_post.device) for filter in self.filters]
        assert self.Omega.shape[1:] == (self.num_filters, self.num_filters)

        P_mix = [torch.zeros_like(filter.P_post) for filter in self.filters]

        for i in range(self.num_filters):
            for j in range(self.num_filters):
                state_mix[i] += self.model_trans[i][j][None,
                                                       ...] @ self.filters[j].state_post * self.Omega[..., j, i].view(
                                                           self.batch_size, 1, 1)

            for j in range(self.num_filters):
                diff = self.model_trans[i][j][None, ...] @ self.filters[j].state_post - state_mix[i]
                P = self.model_trans[i][j][None, ...] @ self.filters[j].P_post @ self.model_trans[i][j][
                    None, ...].transpose(-1, -2) + (diff @ diff.mT)
                P_mix[i] += self.Omega[..., j, i].view(self.batch_size, 1, 1) * P

        # step2: filt
        for j in range(self.num_filters):
            self.filters[j].state_post = state_mix[j].clone()
            self.filters[j].P_post = P_mix[j].clone()
            self.filters[j].predict_step()

        self._compute_state_estimate()

    def update_step(self, obs: torch.Tensor):

        for j in range(self.num_filters):
            self.filters[j].update_step(obs)

        obs_valid = ~torch.isnan(obs).any(dim=tuple(range(1, obs.ndim)))  # (B,)

        # !!!  The original implementation, which is slow but highly readable.
        # Lambda = torch.zeros_like(self.c_bar)
        # for j in range(self.num_filters):
        #     Innov = self.filters[j].Innov
        #     S = self.filters[j].Sk
        #     logpdf = MultivariateNormal(torch.zeros_like(Innov).squeeze(-1), S)
        #     Lambda[:, j, :] = logpdf.log_prob(Innov.squeeze(-1)).exp().unsqueeze(1)

        # Lambda[Lambda == 0] = torch.finfo(Lambda.dtype).eps

        # !!! A vectorized implementation is employed to improve performance.

        Innovs = torch.stack([f.Innov for f in self.filters], dim=1)  # (B, M, dim_obs, 1)
        Sks = torch.stack([f.Sk for f in self.filters], dim=1)  # (B, M, dim_obs, dim_obs)

        try:
            L = torch.linalg.cholesky(Sks)  # assume positive definite
            y = torch.linalg.solve_triangular(L, Innovs, upper=False)
            mahal = (y ** 2).sum(dim=-2, keepdim=True)  # (B, M, 1, 1)
            log_det_S = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1, keepdim=True)  # (B, M, 1)

            log_Lambda = -0.5 * (self.dim_obs * np.log(2 * np.pi) + log_det_S + mahal.squeeze(-1))
            Lambda_raw = log_Lambda.exp()  # (B, M, 1)
        except RuntimeError:
            # Cholesky failed: fallback to uniform or previous prob

            Lambda_raw = torch.ones_like(self.c_bar)  #

        # Numerical safety
        eps = torch.finfo(Lambda_raw.dtype).eps
        Lambda_raw = torch.clamp(Lambda_raw, min=eps)

        numerical_valid = (~torch.isnan(Lambda_raw)) & (~torch.isinf(Lambda_raw))  # (B, M, 1)

        # All models must be valid for the sample to be considered valid.
        sample_valid = numerical_valid.all(dim=1).squeeze(-1)  # (B,)

        # Final valid mask: observation valid and numerical valid
        final_valid = obs_valid & sample_valid  # (B,)

        new_model_prob = torch.zeros_like(self.model_prob)  # (B, M, 1)

        # Have valid samples, then update.
        Lambda_valid = Lambda_raw[final_valid] * self.c_bar[final_valid]  # (V, M, 1)
        Lambda_valid = Lambda_valid / Lambda_valid.sum(dim=1, keepdim=True)
        new_model_prob[final_valid] = Lambda_valid

        # Use prev model prob for invalid samples.
        new_model_prob[~final_valid] = self.model_prob[~final_valid]

        # 更新
        self.model_prob = new_model_prob

        self._compute_mixing_probabilities()
        # compute mixed IMM state and covariance and save posterior estimate
        self._compute_state_estimate()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        assert self._initialized
        self.predict_step()
        self.update_step(obs)

        return self.state_post, self.model_prob

    def _compute_mixing_probabilities(self):
        """Compute the mixing probability for each filter."""
        # transprob.T @ modelprob.T = (modelprob @ transprob), \
        # for convenience, using column vector.

        # !!! The original implementation, which is slow but highly readable.
        # self.c_bar = self.trans_prob.transpose(-1, -2) @ self.model_prob
        # assert self.c_bar.shape == (self.batch_size, self.num_filters, 1)

        # # \Omega_{k-1|k-1}^{ij}=\frac{p_{ij}U_{k-1}^j}{C^j}
        # Omega = torch.zeros_like(self.Omega)
        # for i in range(self.num_filters):
        #     for j in range(self.num_filters):
        #         Omega[..., i, j] = self.trans_prob[..., i, j] * self.model_prob[..., i, 0] / self.c_bar[..., j, 0]
        # self.Omega = Omega

        # !!! A vectorized implementation is employed to improve performance.
        self.c_bar = torch.bmm(self.trans_prob.transpose(-1, -2), self.model_prob)  # (B, M, 1)
        self.Omega = self.trans_prob * self.model_prob / self.c_bar.permute(0, 2, 1)  # (B, M, M)

    def _generic_compute_state_estimate(self):
        """Computes the IMM's mixed state estimate from each filter using the
        the mode probability self.mu to weight the estimates."""

        state_post = torch.zeros(self.batch_size, self.dim_state, 1, device=self.device)
        for i in range(self.num_filters):
            # By default the last filter has the most data dimensions
            state_post = state_post + self.model_trans[-1][i][None, ...] @ self.filters[i].state_post * self.model_prob[
                ..., i, 0].view(self.batch_size, 1, 1)

        assert state_post.shape == (self.batch_size, self.dim_state, 1)

        P_post = torch.zeros(self.batch_size, self.dim_state, self.dim_state, device=self.device)

        for i in range(self.num_filters):
            diff = self.filters[i].state_post - self.model_trans[i][-1][None, ...] @ state_post
            transed_covariance = self.model_trans[-1][i][None, ...] @ self.filters[i].P_post @ self.model_trans[-1][i][
                None, ...].transpose(-1, -2)
            P = self.model_trans[-1][i][None, ...] @ (diff @ diff.mT) @ self.model_trans[-1][i][None, ...].transpose(
                -1, -2) + transed_covariance
            P_post = self.model_prob[..., i, 0].view(self.batch_size, 1, 1) * P + P_post

        self.state_post = state_post
        self.P_post = P_post

    def _fast_compute_state_estimate(self):
        """Fully vectorised only model_trans is SelfTrans, the fastest method
        available, but with one fundamental requirement:

        foremost among them is the requirement that all of the filters in the bank have the same dimensional design.
        """
        X = torch.stack([f.state_post for f in self.filters], dim=1)  # (B, M, d, 1)
        P = torch.stack([f.P_post for f in self.filters], dim=1)  # (B, M, d, d)
        mu = self.model_prob  # (B, M, 1)

        # Mean
        state_post = (X * mu.unsqueeze(-1)).sum(dim=1)  # (B, d, 1)

        # Covariance
        diff = X - state_post.unsqueeze(1)  # (B, M, d, 1)
        outer = diff @ diff.transpose(-1, -2)  # (B, M, d, d)
        P_total = P + outer  # (B, M, d, d)
        P_post = (P_total * mu.unsqueeze(-1)).sum(dim=1)  # (B, d, d)

        self.state_post = state_post
        self.P_post = P_post

    def forward_loop(self, obs: torch.Tensor) -> torch.Tensor:
        N, dim_obs, T = obs.shape
        assert dim_obs == self.dim_obs
        obs = obs.to(self.device)

        self.state_post_seq = torch.zeros(N, self.dim_state, T, device=self.device)
        self.model_probs = torch.zeros(N, self.num_filters, T, device=self.device)

        for t in range(T):
            current_prediction, current_model_prob = self(obs[..., [t]])
            self.state_post_seq[..., [t]] = current_prediction
            self.model_probs[..., [t]] = current_model_prob

        return self.state_post_seq


class MOTIMMFilter(IMMFilter):

    def init_beliefs(self, state: torch.Tensor, covariance: torch.Tensor | None = None) -> None:
        self.batch_size = state.shape[0]
        # assert state.shape == (self.batch_size, self.dim_state, 1)

        # if covariance is None :
        #     covariance = torch.eye(self.dim_state)[None, ...].expand(
        #         self.batch_size, self.dim_state, self.dim_state).clone()

        if covariance is not None:
            assert covariance.shape == (self.batch_size, self.dim_state, self.dim_state)
            covariance = covariance.to(self.device)

        state = state.to(self.device)

        self.model_prob = self._init_model_prob.expand(self.batch_size, self.num_filters, 1).clone()
        self.trans_prob = self._init_trans_prob.expand(self.batch_size, self.num_filters, self.num_filters).clone()
        self.Omega = self._init_Omega.expand_as(self.trans_prob).clone()

        # Use the last filter to initialize the follow-up model.
        for i, filter in enumerate(self.filters):
            if self.trans_init:
                transed_state = self.model_trans[i][-1][None, ...] @ state
            else:
                transed_state = state

            if covariance is not None and self.trans_init:
                transed_covariance = self.model_trans[i][-1][None, ...] @ covariance @ self.model_trans[i][-1][
                    None, ...].transpose(-1, -2)
            elif covariance is not None and not self.trans_init:
                transed_covariance = covariance
            else:
                transed_covariance = None
            filter.init_beliefs(transed_state, transed_covariance)

        self._compute_mixing_probabilities()

        # compute mixed IMM state and covariance and save posterior estimate
        self._compute_state_estimate()

        self._initialized = True
