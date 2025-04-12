from typing import Any

import torch
import torch.nn as nn

from filternet.registry import DYNAMICSMODEL


@DYNAMICSMODEL.register_module()
class LinearDynamicsModel(nn.Module):

    def __init__(self, dim_state: int, device: str = 'cuda') -> None:
        super(LinearDynamicsModel, self).__init__()

        self.dim_state = dim_state
        self.initialized_matrices = False
        self.device = device

    def initialize_matrices(self, params: Any) -> None:
        self.params = params
        assert params.F.shape[-2:] == (self.dim_state, self.dim_state)
        assert params.Q.shape[-2:] == (self.dim_state, self.dim_state)

        self.initialized_matrices = True

    def forward(self, state: torch.Tensor):
        N, dim_state, _ = state.shape
        state = state.to(self.device)
        assert self.initialized_matrices
        assert dim_state == self.dim_state

        pred, jac = self.params.get_pred_jac(state)
        return pred, jac[..., -1], self.params.Q.expand(N, dim_state,
                                                        -1).to(self.device)


# Reference torchfilter https://stanford-iprl-lab.github.io/torchfilter/
@DYNAMICSMODEL.register_module()
class EKFDynamicsModel(LinearDynamicsModel):

    def initialize_matrices(self, params) -> None:
        self.params = params
        assert self.params.F.shape[-2:] == (self.dim_state, self.dim_state)
        assert self.params.Q.shape[-2:] == (self.dim_state, self.dim_state)
        self.initialized_matrices = True

    def forward(self, state: torch.Tensor):
        N, dim_state, _ = state.shape
        state = state.to(self.device)
        assert self.initialized_matrices
        assert dim_state == self.dim_state

        pred, jac = self.params.get_pred_jac(state)
        return pred, jac[..., -1], self.params.Q.expand(N, dim_state,
                                                        -1).to(self.device)
