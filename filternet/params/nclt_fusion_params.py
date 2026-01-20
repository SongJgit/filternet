from typing import List, Tuple
import torch
from filternet.registry import PARAMS
from .base_motion_params import BaseMotionParams


@PARAMS.register_module()
class NCLTFusionWheelGPSParams(BaseMotionParams):

    def __init__(
        self,
        dim_state: int = 6,
        obs_ind: List[int] | None = [0, 1],
        dt: int | float = 1.,
        noise_r2: int | None | float = None,
        noise_q2: int | None | float = None,
    ):
        super().__init__(dim_state=dim_state, obs_ind=obs_ind, dt=dt, noise_r2=noise_r2, noise_q2=noise_q2),
        """NCLT only need velocity parameters

        """
        self.F = torch.eye(self.dim_state)
        H = torch.eye(dim_state)
        self.H = H[obs_ind]
        self.sensor_based = 'wheel'

    def get_pred_jac(self, states, sensor) -> Tuple[torch.Tensor, ...]:
        """_summary_

        Args:
            states (_type_): [bs, [x, y, x', y', theta_k, omega_k],1]
           sensor (_type_):  [bs, [ax , ay , theta, omega],1]
        Returns :
                predict state, and jacobian
        """
        with torch.inference_mode(mode=False):
            batch_size, state_dim, _ = states.shape
            states = states.detach().clone().repeat(1, 1, state_dim)  # for autograd
            sensor = sensor.detach().clone().repeat(1, 1, state_dim)
            states = states.requires_grad_(True)
            sensor = sensor.requires_grad_(True)

            vl_wheel = sensor[:, 0, :]
            vr_wheel = sensor[:, 1, :]
            heading = sensor[:, 2, :]
            omega = sensor[:, 3, :]
            x_k = states[:, 0, :]
            y_k = states[:, 1, :]
            # x_dot_k = states[:, 2, :]
            # y_dot_k = states[:, 3, :]
            # theta_k = states[:, 4, :]
            # omega_k = states[:, 5, :]
            dt_sym = 1

            v_c = (0.5 * (vl_wheel + vr_wheel))
            f1 = (x_k + v_c * torch.cos(heading) * dt_sym)
            f2 = (y_k + v_c * torch.sin(heading) * dt_sym)
            f3 = (v_c * torch.cos(heading))
            f4 = (v_c * torch.sin(heading))
            # x_dot, theta, omgea for auto grad unused,
            # use zero factor ensures that there is no impact on the jac results.
            f5 = heading
            f6 = omega
            f = torch.cat([f1, f2, f3, f4, f5, f6], -1).reshape(batch_size, state_dim, state_dim)

            mask = torch.eye(state_dim).to(f.device).repeat(batch_size, 1, 1)
            jac = torch.autograd.grad(f, states, mask, create_graph=True,
                                      materialize_grads=True)[0].transpose(-1, -2)  # [bs, dim_obs, dim_state, seq_len]
            # jac = torch.zeros_like(jac)
            # jac = torch.zeros(batch_size, state_dim, state_dim).to(f.device)
            # jac[:, 0, 0] = 1
            # jac[:, 1, 1] = 1
            f[:, 4, 0] = wraptopi(f[:, 4, 0])
        return f[:, :, [0]], jac

    def get_obs_jac(self, states: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """GPS Measurement Model is linear, so is H.

        Args:
            states (torch.Tensor): [bs, num_state, 1]

        Returns:
            Tuple[torch.Tensor, ...]: y_pred [bs, num_obs_state, 1], jac
        """
        batch = states.shape[0]
        jac = self.H[None, ...].repeat(batch, 1, 1).to(states.device)
        y_pred = self.H[None, ...].to(states.device) @ states
        return y_pred, jac

    def convert(self, states):
        states[:, 4, :] = wraptopi(states[:, 4, :])
        return states


def wraptopi(x: torch.Tensor) -> torch.Tensor:
    """
    Modified from mte546-project numpy.array to torch.batch.
    Wrap theta measurements to [-pi, pi].
    Accepts an angle measurement in radians and returns an angle measurement in radians
    Args:
        x (torch.Tensor): [bs, 1], [1, bs] or another shape.

    Returns:
        torch.Tensor: equips to x.shape
    """
    pos_pi_mask = x > torch.pi
    neg_pi_mask = x < -torch.pi
    x[pos_pi_mask] = x[pos_pi_mask] - (torch.floor(x[pos_pi_mask] / (2 * torch.pi)) + 1) * 2 * torch.pi
    x[neg_pi_mask] = x[neg_pi_mask] + (torch.floor(x[neg_pi_mask] / (-2 * torch.pi)) + 1) * 2 * torch.pi
    return x
