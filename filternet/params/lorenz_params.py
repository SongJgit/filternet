from typing import Tuple

import torch

from filternet.registry import PARAMS
import math


@PARAMS.register_module()
class LorenzParams:

    def __init__(self,
                 dim_state: int = 3,
                 dim_obs: int = 3,
                 j_test=5,
                 delta_t=0.02):
        self.dim_state = dim_state
        self.dim_obs = dim_obs

        self.B = torch.Tensor([[[0, 0, 0], [0, 0, -1], [0, 1, 0]],
                               torch.zeros(3, 3),
                               torch.zeros(3, 3)]).type(torch.FloatTensor)
        self.C = torch.Tensor([[-10, 10, 0], [28, -1, 0], [0, 0, -8 / 3]])
        self.F = torch.eye(self.dim_state)
        self.j_test = j_test
        self.delta_t = delta_t
        self.H = torch.eye(self.dim_obs, self.dim_state)

    def get_pred_jac(self, state: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa
        device = state.device
        A = torch.einsum('bnl,bnij->bij', state,
                         self.B[None, ...].to(device)) + self.C[None,
                                                                ...].to(device)
        for j in range(1, self.j_test + 1):
            F_add = (torch.matrix_power(A * self.delta_t, j) /
                     math.factorial(j))
            F = torch.add(self.F[None, ...].to(device), F_add)
        return torch.matmul(F, state), None

    def get_obs_jac(self, state: torch.Tensor,
                    **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa
        batch_size, _, seq_len = state.shape

        return state, torch.eye(self.dim_obs,
                                self.dim_state)[None, ..., None].repeat(
                                    batch_size, 1, 1, seq_len).to(state.device)
