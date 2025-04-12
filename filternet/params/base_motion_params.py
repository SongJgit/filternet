"""
Generalized Motion and Measurement Models and Discrete White Noise Motion
model: CT, CA, CV Measurement model: linear measurement model. [1,1]

Discrete White Noise : Gamma @ Gamma.T * variance.Q = {Gamma} @ {Gamma}^{T} * var

"""

from typing import Any, List, Tuple

import torch

from filternet.registry import PARAMS

from .discretization import Q_continuous_white_noise, Q_discrete_white_noise


# mypy: ignore-errors
class BaseMotionParams:

    def __init__(self,
                 dim_state: int,
                 obs_ind: List[int] | None = None,
                 dt: int | float = 1.,
                 noise_r2: int | None | float = None,
                 noise_q2: int | None | float = None,
                 is_discrete_q: bool = True,
                 is_cartesian: bool = True):
        """_summary_

        Args:
            dim_state (int): Number of state variables.
            obs_ind (List[int] | None, optional): Observation index in state, \
                 Ex: state = [x, x', y, y'], if observation is [x, y], then obs_ind = [0, 2]. \
                 len(obs_ind) == observation_dim. Defaults to None.
            dt (int | float, optional): Time interval. Defaults to 1..
            noise_r2 (int | None | float, optional): Variance of measurement noise. Defaults to None.
            noise_q2 (int | None | float, optional): Variance of process noise. Defaults to None.
            is_discrete_q (bool, optional): Whether continuous or discrete q. Defaults to True.
            is_cartesian (bool, optional): Whether cartesian coordinate. Defaults to True.
        """
        self.dim_state = dim_state
        if obs_ind is None:
            self.obs_ind = list(range(dim_state))
        else:
            self.obs_ind = obs_ind
        self.dim_obs = len(self.obs_ind)
        self.dt = dt
        self.half_dt2 = 0.5 * self.dt**2
        self.noise_r2 = noise_r2
        self.noise_q2 = noise_q2
        self.is_discrete_q = is_discrete_q
        self.is_cartesian = is_cartesian

        assert len(self.obs_ind) <= self.dim_state

        self.Q: torch.Tensor
        self.R: torch.Tensor
        self.F: torch.Tensor

    def get(self, name: str) -> Any:
        return getattr(self, name)

    def get_pred_jac(self, state: torch.Tensor, *args):
        raise NotImplementedError

    def get_obs_jac(self, state: torch.Tensor, *args):
        raise NotImplementedError


@PARAMS.register_module()
class CVParams(BaseMotionParams):

    def __init__(self,
                 dim_state: int = 6,
                 obs_ind: List[int] | None = [0, 2, 4],
                 dt: int | float = 1,
                 noise_r2: int | None = None,
                 noise_q2: int | None = None,
                 is_discrete_q: bool = True,
                 is_cartesian: bool = True):
        """Constant velocity model. Support 1 dim , 2 dim and 3 dim,
        crossponding to x, y, z.

        Args:
            dim_state (int): Number of state variables, Must in [2, 4, 6],\
                crossponding to [x, x'], [x, x', y, y'], [x, x', y, y', z, z']. \
                Defaults to 3.
            obs_ind (List[int] | None, optional): Observation index in state, \
                 Ex: state = [x, x', y, y'], if observation is [x, y], then obs_ind = [0, 2]. \
                 len(obs_ind) == observation_dim. Defaults to None.
            dt (int | float, optional): Time interval. Defaults to 1..
            noise_r2 (int | None | float, optional): Variance of measurement noise. Defaults to None.
            noise_q2 (int | None | float, optional): Variance of process noise. Defaults to None.
            is_discrete_q (bool, optional): Whether continuous or discrete q. Defaults to True.
            is_cartesian (bool, optional): Whether cartesian coordinate. Defaults to True.
        """

        super(CVParams, self).__init__(dim_state=dim_state,
                                       obs_ind=obs_ind,
                                       dt=dt,
                                       noise_r2=noise_r2,
                                       noise_q2=noise_q2,
                                       is_discrete_q=is_discrete_q,
                                       is_cartesian=is_cartesian)

        assert self.dim_state in [
            2, 4, 6
        ], (f'CVParams requires at least dimension of state in [2, 4, 6],'
            f"corresponding to [x, x'], [x, x', y, y'], [x, x', y, y', z, z'],"
            f'but {self.dim_state=}')

        # corresponds to [x, x', y, y', z, z']
        F = torch.tensor([[
            1.,
            self.dt,
        ], [0., 1.]])
        self.F = torch.kron(torch.eye(self.dim_state // 2), F)

        H = torch.eye(dim_state)
        self.H = H[obs_ind]

        if self.is_discrete_q:
            # Q = \Gamma @ \Gamma.T @ noise_q2
            Q = torch.from_numpy(
                Q_discrete_white_noise(2,
                                       dt=self.dt,
                                       var=self.noise_q2,
                                       block_size=self.dim_state // 2)).to(
                                           torch.float32)
        else:
            Q = torch.from_numpy(
                Q_continuous_white_noise(2,
                                         dt=self.dt,
                                         spectral_density=self.noise_q2,
                                         block_size=self.dim_state // 2)).to(
                                             torch.float32)
        self.Q = Q

        R = torch.eye(len(obs_ind)) * self.noise_r2
        self.R = R

        self.P = torch.eye(self.dim_state)

    def get_pred_jac(self, state: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa
        """_summary_

        Args:
            state (torch.Tensor): [batch_size, dim_state, seq_len].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: results of F@State and Jacobian of F@State, \
                [bs, dim_state, seq_len] and [bs, dim_state, dim_state, seq_len]
        """
        batch_size = state.shape[0]
        out = self.F[None, ...].to(state.device) @ state
        jac = self.F[None, ...].repeat(batch_size, 1, 1).to(state.device)
        return out, jac[..., None]

    def get_obs_jac(self, state: torch.Tensor,
                    **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa
        """

        Args:
            state (torch.Tensor): [batch_size, dim_state, seq_len]

        Raises:
            NotImplementedError: _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: results of H@State and Jacobian of H@State.\
            [bs, dim_obs, seq_len] and [bs, dim_obs, dim_state, seq_len]
        """
        batch_size, dim_state, seq_len = state.shape
        assert self.dim_state == dim_state, f'{self.dim_state=} != {dim_state=}'

        if self.is_cartesian or dim_state == 2:
            out = self.H[None, ...].to(state.device) @ state
            jac = self.H[None, ..., None].repeat(batch_size, 1, 1,
                                                 seq_len).to(state.device)

        elif dim_state == 4:  # [x, x', y, y']
            x = state[:, [0], :]
            y = state[:, [2], :]
            r = torch.linalg.norm(state[:, [0, 2], :], keepdim=True, dim=1)
            phi = torch.atan2(y, x)
            out = torch.cat([r, phi], dim=1)

            with torch.inference_mode(mode=False):
                # in test_step, pl will be in eval mode,
                # so we need to set it to False mode use torch.autograd.grad() to compute jacobian.
                _state = state.clone().detach().requires_grad_(True)
                _state = _state.unsqueeze(-2).repeat(
                    1, 1, self.dim_obs,
                    1)  # -> [bs, dim_state, dim_obs, seq_len]
                _x = _state[:, [0], :, :]
                _y = _state[:, [2], :, :]
                _r = torch.linalg.norm(_state[:, [0, 3], :, :],
                                       keepdim=True,
                                       dim=1)
                _phi = torch.atan2(_y, _x)
                _out = torch.cat([_r, _phi],
                                 dim=1).expand(batch_size, self.dim_obs,
                                               self.dim_obs, seq_len)
                mask = torch.eye(self.dim_obs).unsqueeze(-1).repeat(
                    batch_size, 1, 1, seq_len).to(
                        _state.device)  # [bs, dim_obs, dim_obs, seq_len]
                jac = torch.autograd.grad(
                    _out,
                    _state,
                    mask,
                    create_graph=True,
                    materialize_grads=True)[0].transpose(
                        -2, -3)  # [bs, dim_obs, dim_state, seq_len]
        elif dim_state == 6:  # [x, x', y, y', z, z']
            x = state[:, [0], :]
            y = state[:, [2], :]
            z = state[:, [4], :]
            r = torch.linalg.norm(state[:, [0, 2, 4], :], keepdim=True, dim=1)
            theta = torch.acos(torch.div(z, r))
            phi = torch.atan2(y, x)
            out = torch.cat([r, theta, phi], dim=1)

            _state = state.clone().detach()
            with torch.inference_mode(mode=False):
                _state = _state.unsqueeze(-2).repeat(
                    1, 1, self.dim_obs, 1).requires_grad_(
                        True)  # -> [bs, dim_state, dim_obs, seq_len]
                _x = _state[:, [0], :, :]
                _y = _state[:, [2], :, :]
                _z = _state[:, [4], :, :]
                _r = torch.linalg.norm(
                    _state[:, [0, 3, 6], :, :], keepdim=True,
                    dim=1)  # equal to sqrt(x**2 + y**2 + z**2), but stability.
                _theta = torch.acos(torch.div(_z, _r))
                _phi = torch.atan2(_y, _x)
                _out = torch.cat([_r, _theta, _phi],
                                 dim=1).expand(batch_size, self.dim_obs,
                                               self.dim_obs, seq_len)
                mask = torch.eye(self.dim_obs).unsqueeze(-1).repeat(
                    batch_size, 1, 1, seq_len).to(
                        _state.device)  # [bs, dim_obs, dim_obs, seq_len]
                jac = torch.autograd.grad(
                    _out,
                    _state,
                    mask,
                    create_graph=True,
                    materialize_grads=True)[0].transpose(
                        -2, -3)  # [bs, dim_obs, dim_state, seq_len]
        else:
            raise NotImplementedError(
                f'{self.dim_state=} is not supported, only support 4 and 6')
        return out, jac


@PARAMS.register_module()
class CAParams(BaseMotionParams):

    def __init__(self,
                 dim_state: int = 6,
                 obs_ind: List[int] | None = [0, 2, 4],
                 dt: int | float = 1,
                 noise_r2: int | None = None,
                 noise_q2: int | None = None,
                 is_discrete_q: bool = True,
                 is_cartesian: bool = True):
        """Constant accelerate model. Support 1 dim , 2 dim and 3 dim,
        crossponding to x, y, z.

        Args:
            dim_state (int): Number of state variables, Must in [3, 6, 9],\
                crossponding to [x, x', x''], [x, x', x'', y, y', y''], [x, x', x'', y, y', y'', z, z', z'']. \
                Defaults to 9.
            obs_ind (List[int] | None, optional): Observation index in state, \
                Must in [0], [0, 3], [0, 3, 6], crossponding to [x], [x, y], [x, y, z]. \
                Ex: state = [x, x', y, y'], if observation is [x, y], then obs_ind = [0, 2]. \
                len(obs_ind) == observation_dim. Defaults to None.
            dt (int | float, optional): Time interval. Defaults to 1..
            noise_r2 (int | None | float, optional): Variance of measurement noise. Defaults to None.
            noise_q2 (int | None | float, optional): Variance of process noise. Defaults to None.
            is_discrete_q (bool, optional): Whether continuous or discrete q. Defaults to True.
            is_cartesian (bool, optional): Whether cartesian coordinate. Defaults to True.
        """

        super(CAParams, self).__init__(dim_state=dim_state,
                                       obs_ind=obs_ind,
                                       dt=dt,
                                       noise_r2=noise_r2,
                                       noise_q2=noise_q2,
                                       is_discrete_q=is_discrete_q,
                                       is_cartesian=is_cartesian)

        assert self.dim_state in [
            3, 6, 9
        ], (f'CAParams requires at least state dimension >= 3 and at must <=9,'
            f"corresponding to [x, x', x''], [x, x', x'', y, y', y''], [x, x', x'', y, y', y'', z, z', z''],"
            f'but {self.dim_state=}')
        # corresponds to [x, x', x'' y, y', y'', z, z', z'']
        F = torch.tensor([[1., self.dt, self.half_dt2], [0., 1., self.dt],
                          [0., 0., 1]])
        self.F = torch.kron(torch.eye(self.dim_state // 3), F)

        if self.is_discrete_q:
            # Q = \Gamma @ \Gamma.T @ noise_q2
            Q = torch.from_numpy(
                Q_discrete_white_noise(3,
                                       dt=self.dt,
                                       var=self.noise_q2,
                                       block_size=self.dim_state // 3)).to(
                                           torch.float32)
        else:
            Q = torch.from_numpy(
                Q_continuous_white_noise(3,
                                         dt=self.dt,
                                         spectral_density=self.noise_q2,
                                         block_size=self.dim_state // 3)).to(
                                             torch.float32)
        self.Q = Q

        H = torch.eye(dim_state)
        self.H = H[obs_ind]

        R = torch.eye(len(obs_ind)) * self.noise_r2
        self.R = R

        self.P = torch.eye(self.dim_state)

    def get_pred_jac(self, state: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa
        """_summary_

        Args:
            state (torch.Tensor): [batch_size, dim_state, seq_len].

        Raises:
            NotImplementedError: _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: results of H@State and Jacobian of H@State.\
            [bs, dim_obs, seq_len] and [bs, dim_obs, dim_state, seq_len]
        """
        batch_size = state.shape[0]
        out = self.F[None, ...].to(state.device) @ state
        jac = self.F[None, ...].repeat(batch_size, 1, 1).to(state.device)
        return out, jac[..., None]

    def get_obs_jac(self, state: torch.Tensor,
                    **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa
        """_summary_

        Args:
            state (torch.Tensor): [batch_size, dim_state, seq_len].

        Raises:
            NotImplementedError: _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: results of H@State and Jacobian of H@State.\
            [bs, dim_obs, seq_len] and [bs, dim_obs, dim_state, seq_len]
        """

        batch_size, dim_state, seq_len = state.shape
        assert dim_state == self.dim_state, f'{dim_state=} != {self.dim_state=}'
        if self.is_cartesian or dim_state == 3:
            out = self.H[None, ...].to(state.device) @ state
            jac = self.H[None, ..., None].repeat(batch_size, 1, 1,
                                                 seq_len).to(state.device)

        elif dim_state == 6:  # [x, x',x'', y, y',y'']
            x = state[:, [0], :]
            y = state[:, [3], :]
            r = torch.linalg.norm(state[:, [0, 3], :], keepdim=True, dim=1)
            phi = torch.atan2(y, x)
            out = torch.cat([r, phi], dim=1)
            with torch.inference_mode(mode=False):
                _state = state.clone().detach().requires_grad_(True)
                _state = _state.unsqueeze(-2).repeat(
                    1, 1, self.dim_obs,
                    1)  # -> [bs, dim_state, dim_obs, seq_len]
                _x = _state[:, [0], :, :]
                _y = _state[:, [3], :, :]
                _r = torch.linalg.norm(
                    _state[:, [0, 3], :, :], keepdim=True,
                    dim=1)  # equal to sqrt(x**2 + y**2 + z**2), but stability.
                _phi = torch.atan2(_y, _x)
                _out = torch.cat([_r, _phi],
                                 dim=1).expand(batch_size, self.dim_obs,
                                               self.dim_obs, seq_len)
                mask = torch.eye(self.dim_obs).unsqueeze(-1).repeat(
                    batch_size, 1, 1, seq_len).to(
                        _state.device)  # [bs, dim_obs, dim_obs, seq_len]
                jac = torch.autograd.grad(
                    _out,
                    _state,
                    mask,
                    create_graph=True,
                    materialize_grads=True)[0].transpose(
                        -2, -3)  # [bs, dim_obs, dim_state, seq_len]
        elif self.dim_state == 9:  # [x, x', x'', y, y', y'', z, z', z'']
            x = state[:, [0], :]
            y = state[:, [3], :]
            z = state[:, [6], :]
            r = torch.linalg.norm(state[:, [0, 3, 6], :], keepdim=True, dim=1)
            theta = torch.acos(torch.div(z, r))
            phi = torch.atan2(y, x)
            out = torch.cat([r, theta, phi], dim=1)

            with torch.inference_mode(mode=False):
                _state = state.clone().detach().requires_grad_(True)
                _state = _state.unsqueeze(-2).repeat(
                    1, 1, self.dim_obs,
                    1)  # -> [bs, dim_state, dim_obs, seq_len]
                _x = _state[:, [0], :, :]
                _y = _state[:, [3], :, :]
                _z = _state[:, [6], :, :]
                _r = torch.linalg.norm(
                    _state[:, [0, 3, 6], :, :], keepdim=True,
                    dim=1)  # equal to sqrt(x**2 + y**2 + z**2), but stability.
                _theta = torch.acos(torch.div(_z, _r))
                _phi = torch.atan2(_y, _x)
                _out = torch.cat([_r, _theta, _phi],
                                 dim=1).expand(batch_size, self.dim_obs,
                                               self.dim_obs, seq_len)
                mask = torch.eye(self.dim_obs).unsqueeze(-1).repeat(
                    batch_size, 1, 1, seq_len).to(
                        _state.device)  # [bs, dim_obs, dim_obs, seq_len]
                jac = torch.autograd.grad(
                    _out,
                    _state,
                    mask,
                    create_graph=True,
                    materialize_grads=True)[0].transpose(
                        -2, -3)  # [bs, dim_obs, dim_state, seq_len]
        return out, jac


# TODO: add remains
class CTParams(BaseMotionParams):

    def __init__(
        self,
        dim_state: int = 4,
        obs_ind: List[int] | None = [0, 2],
        T: int | float = 1,
        theta: int | float = 5,
        noise_r2: int | None = None,
        noise_q2: int | None = None,
        is_cartesian: bool = False,
        #  dataset: Dict[str, Any] | Dataset | None = None
    ):
        """ CT Models with Unknown Turn Rate. so like state = [x, x', omega].

        Args:
            dim_state (int, optional): _description_. Defaults to 4.
            obs_ind (List[int] | None, optional): _description_. Defaults to [0, 2].
            T (int, optional): _description_. Defaults to 1.
            theta (int, optional): _description_. Defaults to 5.
            noise_r2 (int | None, optional): _description_. Defaults to None.
            noise_q2 (int | None, optional): _description_. Defaults to None.
            constant_noise (bool, optional): _description_. Defaults to True.
            dataset (Dict[str, Any] | Dataset | None, optional): _description_. Defaults to None.
        """
        super(CTParams, self).__init__(dim_state, obs_ind, T, noise_r2,
                                       noise_q2)

        # theta : turn rate
        # [x, x', y, y']
        theta = torch.pi / 180 * theta
        self.thetaT = torch.tensor(theta * self.dt, dtype=torch.float32)
        F = torch.tensor([
            [
                1,
                torch.sin(self.thetaT) / theta, 0,
                (torch.cos(self.thetaT) - 1) / theta, 0.
            ],
            [0., torch.cos(self.thetaT), 0., -torch.sin(self.thetaT), 0.],
            [  # noqa
                0.,
                (1 - torch.cos(self.thetaT)) / theta,
                1.,  # noqa
                torch.sin(self.thetaT) / theta,
                0.  # noqa
            ],  # noqa
            [0., torch.sin(self.thetaT), 0.,
             torch.cos(self.thetaT), 0.],
            [0., 0., 0., 0., 1.]
        ])
        self.F = F
        Gamma = torch.tensor([[self.half_dt2], [self.dt]])
        Gamma = torch.kron(torch.eye(dim_state // 2), Gamma)
        self.Gamma = Gamma
        Q = self.Gamma @ self.Gamma.T * self.noise_q2
        self.Q = Q
        H = torch.eye(dim_state)
        self.H = H[obs_ind]
