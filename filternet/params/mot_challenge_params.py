from typing import List, Tuple

import torch

from filternet.registry import PARAMS


class MOTChallengeParams:

    def __init__(self,
                 dim_state: int = 8,
                 obs_ind: List[int] | None = [0, 2, 4, 6],
                 dt: int | float = 1,
                 std_weight_position: float = 1. / 20,
                 std_weight_velocity: float = 1. / 160,
                 std_weight_acceleration: float = 1. / 160,
                 update_noise: bool = False):

        self.dt = dt
        self.dim_state = dim_state
        self.obs_ind = obs_ind
        self.dim_obs = len(self.obs_ind)
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity
        self._std_weight_acceleration = std_weight_acceleration
        self.update_noise = update_noise

        self.half_dt2 = 0.5 * self.dt**2

    def get_pred_jac(self,
                     state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.update_noise:
            self._update_Q(state)

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
        if self.update_noise:
            self._update_R(state)
        batch_size, dim_state, seq_len = state.shape
        assert self.dim_state == dim_state, f'{self.dim_state=} != {dim_state=}'

        out = self.H[None, ...].to(state.device) @ state
        jac = self.H[None, ..., None].repeat(batch_size, 1, 1,
                                             seq_len).to(state.device)
        return out, jac


@PARAMS.register_module()
class CVMOTXYAHParams(MOTChallengeParams):

    def __init__(
        self,
        dim_state: int = 8,
        obs_ind: List[int] | None = [0, 2, 4, 6],
        dt: int | float = 1,
        std_weight_position: float = 1. / 20,
        std_weight_velocity: float = 1. / 160,
        update_noise: bool = False,
    ):
        super().__init__(dim_state=dim_state,
                         obs_ind=obs_ind,
                         dt=dt,
                         std_weight_position=std_weight_position,
                         std_weight_velocity=std_weight_velocity,
                         update_noise=update_noise)
        # corresponds to [x, x', y, y', w/h, w/h', h, h']

        F = torch.tensor([[
            1.,
            self.dt,
        ], [0., 1.]])

        self.F = torch.kron(torch.eye(self.dim_state // 2), F)

        H = torch.eye(dim_state)
        self.H = H[obs_ind]

        Q = torch.tensor([
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_position,
            self._std_weight_velocity,
            1e-2,
            1e-5,
            self._std_weight_position,
            self._std_weight_velocity,
        ])
        self.Q = torch.diag(Q**2)

        R = torch.tensor([
            self._std_weight_position, self._std_weight_position, 1e-1,
            self._std_weight_position
        ])

        self.R = torch.diag(R**2)

    def initiate(self, state: torch.Tensor,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            state (torch.Tensor): Bounding box coordinates (x, y, a, h) with
            center position (x, y), aspect ratio a, and height h.
            [bs, num_state, 1]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the mean vector (8 dimensional) and
                covariance matrix (8x8 dimensional) of the new track.
                Unobserved velocities are initialized to 0 mean.
                correspond to   [x, x', y, y', w/h, w/h', h, h']
        """
        N, dim_state, _ = state.shape
        state_vel = torch.zeros_like(state)
        new_state = torch.cat([state, state_vel], dim=-1).reshape(
            N, self.dim_state, 1)  # [x, x', y, y', w/h, w/h', h, h']

        P = torch.tensor([
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
            1e-2,
            1e-5,
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
        ]).to(state.device)
        P = torch.diag(P**2)
        _dynamic = state[:, 3, :]  # [bs, 1]
        _dynamic = torch.cat([
            _dynamic,
            _dynamic,
            _dynamic,
            _dynamic,
            torch.ones_like(_dynamic),
            torch.ones_like(_dynamic),
            _dynamic,
            _dynamic,
        ],
                             dim=1).to(state.device)  # noqa: E126
        _dynamic_2 = torch.square(torch.diag_embed(_dynamic))
        self.P = P * _dynamic_2
        self._reset_noise()
        self.Q = self._update_Q(new_state)
        self.R = self._update_R(new_state)

        return new_state, self.P

    def _update_Q(self, state: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            state (torch.Tensor): [bs, [x, x', y, y', w/h, w/h', h, h'],1]

        Returns:
            torch.Tensor: The 8x8 dimensional covariance matrix.
        """
        self._reset_noise()
        batch_size, dim_state, _ = state.shape
        _dynamic = state[:, 6, :]  # [bs ,1]
        _dynamic = torch.cat([
            _dynamic,
            _dynamic,
            _dynamic,
            _dynamic,
            torch.ones_like(_dynamic),
            torch.ones_like(_dynamic),
            _dynamic,
            _dynamic,
        ],
                             dim=1)  # noqa: E126
        if self.Q.ndim < 3:
            self.Q.unsqueeze_(0)  # [bs, ...]
        self.Q = self.Q.to(state.device) * torch.square(
            torch.diag_embed(_dynamic))

        return self.Q

    def _update_R(self, state: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            state (torch.Tensor): [bs, [x, x', y, y', w/h, w/h', h, h'],1]

        Returns:
            torch.Tensor: _description_
        """
        self._reset_noise()
        batch_size, dim_state, _ = state.shape
        _dynamic = state[:, 6, :]  # [bs ,1]
        _dynamic = torch.cat(
            [_dynamic, _dynamic,
             torch.ones_like(_dynamic), _dynamic], dim=1)
        if self.R.ndim < 3:
            self.R.unsqueeze_(0)
        self.R = self.R.to(state.device) * torch.square(
            torch.diag_embed(_dynamic))
        return self.R

    def _reset_noise(self):
        Q = torch.tensor([
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_position,
            self._std_weight_velocity,
            1e-2,
            1e-5,
            self._std_weight_position,
            self._std_weight_velocity,
        ])
        self.Q = torch.diag(Q**2)

        R = torch.tensor([
            self._std_weight_position, self._std_weight_position, 1e-1,
            self._std_weight_position
        ])

        self.R = torch.diag(R**2)


@PARAMS.register_module()
class CAMOTXYAHParams(MOTChallengeParams):

    def __init__(
        self,
        dim_state: int = 12,
        obs_ind: List[int] | None = [0, 3, 6, 9],
        dt: int | float = 1,
        std_weight_position: float = 1. / 20,
        std_weight_velocity: float = 1. / 160,
        std_weight_acceleration: float = 1. / 160,
        update_noise: bool = False,
    ):
        super().__init__(dim_state=dim_state,
                         obs_ind=obs_ind,
                         dt=dt,
                         std_weight_position=std_weight_position,
                         std_weight_velocity=std_weight_velocity,
                         std_weight_acceleration=std_weight_acceleration,
                         update_noise=update_noise)
        # corresponds to [x, x', x'', y, y', y'', w/h, w/h', w/h'', h, h', h'',]

        F = torch.tensor([[1., self.dt, self.half_dt2], [0., 1., self.dt],
                          [0., 0., 1]])
        self.F = torch.kron(torch.eye(self.dim_state // 3), F)

        H = torch.eye(dim_state)
        self.H = H[obs_ind]

        Q = torch.tensor([
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_acceleration,
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_acceleration,
            1e-2,
            1e-5,
            1e-5,
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_acceleration,
        ])
        self.Q = torch.square(torch.diag(Q))

        R = torch.tensor([
            self._std_weight_position, self._std_weight_position, 1e-1,
            self._std_weight_position
        ])

        self.R = torch.diag(R**2)

    def initiate(self, state: torch.Tensor,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            state (torch.Tensor): Bounding box coordinates (x, y, a, h) with
            center position (x, y), aspect ratio a, and height h.
            [bs, num_state, 1]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the mean vector (8 dimensional) and
                covariance matrix (8x8 dimensional) of the new track.
                Unobserved velocities are initialized to 0 mean.
                correspond to   [x, x', x'', y, y', y'', w/h, w/h', w/h'', h, h', h'']
        """
        N, dim_state, _ = state.shape
        state_vel = torch.zeros_like(state)
        state_acc = torch.zeros_like(state)
        new_state = torch.cat([state, state_vel, state_acc], dim=-1).reshape(
            N, self.dim_state,
            1)  # [x, x', x'', y, y', y'', w/h, w/h', w/h'', h, h', h'']

        P = torch.tensor([
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
            10 * self._std_weight_acceleration,
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
            10 * self._std_weight_acceleration,
            1e-2,
            1e-5,
            1e-5,
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
            10 * self._std_weight_acceleration,
        ]).to(state.device)
        P = torch.diag(P**2)
        _dynamic = state[:, 3, :]  # [bs, 1]
        _dynamic = torch.cat([
            _dynamic,
            _dynamic,
            _dynamic,
            _dynamic,
            _dynamic,
            _dynamic,
            torch.ones_like(_dynamic),
            torch.ones_like(_dynamic),
            torch.ones_like(_dynamic),
            _dynamic,
            _dynamic,
            _dynamic,
        ],
                             dim=1)  # noqa: E126
        _dynamic_2 = torch.square(torch.diag_embed(_dynamic))
        self.P = P * _dynamic_2

        self._reset_noise()
        self.Q = self._update_Q(new_state)
        self.R = self._update_R(new_state)
        return new_state, self.P

    def _update_Q(self, state: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            state (torch.Tensor): [bs, [x, x', x'', y, y', y'', w/h, w/h', w/h'', h, h', h''],1]

        Returns:
            torch.Tensor: The 8x8 dimensional covariance matrix.
        """
        self._reset_noise()
        batch_size, dim_state, _ = state.shape
        _dynamic = state[:, 9, :]  # [bs ,1]
        _dynamic = torch.cat([
            _dynamic,
            _dynamic,
            _dynamic,
            _dynamic,
            _dynamic,
            _dynamic,
            torch.ones_like(_dynamic),
            torch.ones_like(_dynamic),
            torch.ones_like(_dynamic),
            _dynamic,
            _dynamic,
            _dynamic,
        ],
                             dim=1)  # noqa: E126
        if self.Q.ndim < 3:
            self.Q.unsqueeze_(0)
        self.Q = self.Q.to(state.device) * torch.square(
            torch.diag_embed(_dynamic))

        return self.Q

    def _update_R(self, state: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            state (torch.Tensor): [bs, [x, x', x'', y, y', y'', w/h, w/h', w/h'', h, h', h''],1]

        Returns:
            torch.Tensor: _description_
        """
        self._reset_noise()
        batch_size, dim_state, _ = state.shape
        _dynamic = state[:, 9, :]  # [bs ,1]
        _dynamic = torch.cat(
            [_dynamic, _dynamic,
             torch.ones_like(_dynamic), _dynamic], dim=1)
        if self.R.ndim < 3:
            self.R.unsqueeze_(0)
        self.R = self.R.to(state.device) * torch.square(
            torch.diag_embed(_dynamic))
        return self.R

    def _reset_noise(self):
        Q = torch.tensor([
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_acceleration,
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_acceleration,
            1e-2,
            1e-5,
            1e-5,
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_acceleration,
        ])
        self.Q = torch.square(torch.diag(Q))

        R = torch.tensor([
            self._std_weight_position, self._std_weight_position, 1e-1,
            self._std_weight_position
        ])

        self.R = torch.diag(R**2)


@PARAMS.register_module()
class CVMOTXYWHParams(MOTChallengeParams):

    def __init__(self,
                 dim_state: int = 8,
                 obs_ind: List[int] | None = [0, 2, 4, 6],
                 dt: int | float = 1,
                 std_weight_position: float = 1. / 20,
                 std_weight_velocity: float = 1. / 160,
                 update_noise: bool = False):
        super().__init__(dim_state=dim_state,
                         obs_ind=obs_ind,
                         dt=dt,
                         std_weight_position=std_weight_position,
                         std_weight_velocity=std_weight_velocity,
                         update_noise=update_noise)
        # corresponds to [x, x', y, y', w, w', h, h']

        F = torch.tensor([[
            1.,
            self.dt,
        ], [0., 1.]])

        self.F = torch.kron(torch.eye(self.dim_state // 2), F)

        H = torch.eye(dim_state)
        self.H = H[obs_ind]

        Q = torch.tensor([
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_position,
            self._std_weight_velocity,
        ])
        self.Q = torch.diag(Q**2)

        R = torch.tensor([
            self._std_weight_position, self._std_weight_position,
            self._std_weight_position, self._std_weight_position
        ])

        self.R = torch.diag(R**2)

    def initiate(self, state: torch.Tensor,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            state (torch.Tensor): Bounding box coordinates (x, y, w, h) with
            center position (x, y), aspect ratio a, and height h.
            [bs, num_state, 1]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the mean vector (8 dimensional) and
                covariance matrix (8x8 dimensional) of the new track.
                Unobserved velocities are initialized to 0 mean.
                correspond to   [x, x', y, y', w/h, w/h', h, h']
        """
        N, dim_state, _ = state.shape
        state_vel = torch.zeros_like(state)
        new_state = torch.cat([state, state_vel], dim=-1).reshape(
            N, self.dim_state, 1)  # [x, x', y, y', w, w', h, h']

        P = torch.tensor([
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
        ]).to(state.device)
        P = torch.diag(P**2)
        w = state[:, 2, :]
        h = state[:, 3, :]  # [bs, 1]
        _dynamic = torch.cat([w, w, h, h, w, w, h, h], dim=1)  # noqa: E126
        _dynamic_2 = torch.square(torch.diag_embed(_dynamic))
        self.P = P * _dynamic_2

        self._reset_noise()
        self.Q = self._update_Q(new_state)
        self.R = self._update_R(new_state)
        return new_state, self.P

    def _update_Q(self, state: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            state (torch.Tensor): [bs, [x, x', y, y', w, w', h, h'],1]

        Returns:
            torch.Tensor: The 8x8 dimensional covariance matrix.
        """
        self._reset_noise()
        batch_size, dim_state, _ = state.shape
        w = state[:, 4, :]  # [bs ,1]
        h = state[:, 6, :]  # [bs ,1]
        _dynamic = torch.cat([
            w,
            w,
            h,
            h,
            w,
            w,
            h,
            h,
        ], dim=1)  # noqa: E126
        if self.Q.ndim < 3:
            self.Q.unsqueeze_(0)
        self.Q = self.Q.to(state.device) * torch.square(
            torch.diag_embed(_dynamic))

        return self.Q

    def _update_R(self, state: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            state (torch.Tensor): [bs, [x, x', y, y', w, w', h, h'],1]

        Returns:
            torch.Tensor: _description_
        """
        self._reset_noise()
        batch_size, dim_state, _ = state.shape
        w = state[:, 4, :]  # [bs ,1]
        h = state[:, 6, :]  # [bs ,1]
        _dynamic = torch.cat([w, h, w, h], dim=1)
        if self.R.ndim < 3:
            self.R.unsqueeze_(0)
        self.R = self.R.to(state.device) * torch.square(
            torch.diag_embed(_dynamic))
        return self.R

    def _reset_noise(self):
        Q = torch.tensor([
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_position,
            self._std_weight_velocity,
        ])
        self.Q = torch.diag(Q**2)

        R = torch.tensor([
            self._std_weight_position, self._std_weight_position,
            self._std_weight_position, self._std_weight_position
        ])

        self.R = torch.diag(R**2)


@PARAMS.register_module()
class CAMOTXYWHParams(MOTChallengeParams):

    def __init__(self,
                 dim_state: int = 12,
                 obs_ind: List[int] | None = [0, 3, 6, 9],
                 dt: int | float = 1,
                 std_weight_position: float = 1. / 20,
                 std_weight_velocity: float = 1. / 160,
                 std_weight_acceleration: float = 1. / 160,
                 update_noise: bool = False):
        super().__init__(dim_state=dim_state,
                         obs_ind=obs_ind,
                         dt=dt,
                         std_weight_position=std_weight_position,
                         std_weight_velocity=std_weight_velocity,
                         std_weight_acceleration=std_weight_acceleration,
                         update_noise=update_noise)
        # corresponds to [x, x', x'', y, y', y'', w, w', w'', h, h', h'',]

        F = torch.tensor([[1., self.dt, self.half_dt2], [0., 1., self.dt],
                          [0., 0., 1]])
        self.F = torch.kron(torch.eye(self.dim_state // 3), F)

        H = torch.eye(dim_state)
        self.H = H[obs_ind]

        Q = torch.tensor([
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_acceleration,
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_acceleration,
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_acceleration,
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_acceleration,
        ])
        self.Q = torch.square(torch.diag(Q))

        R = torch.tensor([
            self._std_weight_position, self._std_weight_position,
            self._std_weight_position, self._std_weight_position
        ])

        self.R = torch.diag(R**2)

    def initiate(self, state: torch.Tensor,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            state (torch.Tensor): Bounding box coordinates (x, y, w, h) with
            center position (x, y), aspect ratio a, and height h.
            [bs, num_state, 1]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the mean vector (8 dimensional) and
                covariance matrix (8x8 dimensional) of the new track.
                Unobserved velocities are initialized to 0 mean.
                correspond to   [x, x', x'', y, y', y'', w/h, w/h', w/h'', h, h', h'']
        """
        N, dim_state, _ = state.shape
        state_vel = torch.zeros_like(state)
        state_acc = torch.zeros_like(state)
        new_state = torch.cat([state, state_vel, state_acc], dim=-1).reshape(
            N, self.dim_state,
            1)  # [x, x', x'', y, y', y'', w, w', w'', h, h', h'']

        P = torch.tensor([
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
            10 * self._std_weight_acceleration,
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
            10 * self._std_weight_acceleration,
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
            10 * self._std_weight_acceleration,
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
            10 * self._std_weight_acceleration,
        ]).to(state.device)
        P = torch.diag(P**2)
        w = state[:, 2, :]
        h = state[:, 3, :]  # [bs, 1]
        _dynamic = torch.cat([
            w,
            w,
            w,
            h,
            h,
            h,
            w,
            w,
            w,
            h,
            h,
            h,
        ], dim=1)  # noqa: E126
        _dynamic_2 = torch.square(torch.diag_embed(_dynamic))
        self.P = P * _dynamic_2
        self._reset_noise()
        self.Q = self._update_Q(new_state)
        self.R = self._update_R(new_state)
        return new_state, self.P

    def _update_Q(self, state: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            state (torch.Tensor): [bs, [x, x', x'', y, y', y'', w, w', w'', h, h', h''],1]

        Returns:
            torch.Tensor: The 8x8 dimensional covariance matrix.
        """
        self._reset_noise()
        batch_size, dim_state, _ = state.shape
        w = state[:, 6, :]
        h = state[:, 9, :]
        _dynamic = torch.cat([
            w,
            w,
            w,
            h,
            h,
            h,
            w,
            w,
            w,
            h,
            h,
            h,
        ], dim=1)  # noqa: E126
        if self.Q.ndim < 3:
            self.Q.unsqueeze_(0)
        self.Q = self.Q.to(state.device) * torch.square(
            torch.diag_embed(_dynamic))

        return self.Q

    def _update_R(self, state: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            state (torch.Tensor): [bs, [x, x', x'', y, y', y'', w, w, w'', h, h', h''],1]

        Returns:
            torch.Tensor: _description_
        """
        self._reset_noise()
        batch_size, dim_state, _ = state.shape
        w = state[:, 6, :]
        h = state[:, 9, :]
        _dynamic = torch.cat([w, h, w, h], dim=1)
        if self.R.ndim < 3:
            self.R.unsqueeze_(0)
        self.R = self.R.to(state.device) * torch.square(
            torch.diag_embed(_dynamic))
        return self.R

    def _reset_noise(self):
        Q = torch.tensor([
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_acceleration,
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_acceleration,
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_acceleration,
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_acceleration,
        ])
        self.Q = torch.square(torch.diag(Q))

        R = torch.tensor([
            self._std_weight_position, self._std_weight_position,
            self._std_weight_position, self._std_weight_position
        ])

        self.R = torch.diag(R**2)


class STMOTXYWHParams(MOTChallengeParams):

    def __init__(self,
                 dim_state: int = 4,
                 obs_ind: List[int] | None = [0, 1, 2, 3],
                 dt: int | float = 1,
                 std_weight_position: float = 1. / 20,
                 update_noise: bool = False):
        super().__init__(dim_state=dim_state,
                         obs_ind=obs_ind,
                         dt=dt,
                         std_weight_position=std_weight_position,
                         update_noise=update_noise)
        # corresponds to [x, x', x'', y, y', y'', w, w', w'', h, h', h'',]

        F = torch.tensor([[1.]])
        self.F = torch.kron(torch.eye(self.dim_state), F)
        H = torch.eye(dim_state)
        self.H = H[obs_ind]
        Q = torch.tensor([
            self._std_weight_position,
            self._std_weight_position,
            self._std_weight_position,
            self._std_weight_position,
        ])
        self.Q = torch.square(torch.diag(Q))

        R = torch.tensor([
            self._std_weight_position, self._std_weight_position,
            self._std_weight_position, self._std_weight_position
        ])

        self.R = torch.diag(R**2)

    def initiate(self, state: torch.Tensor,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            state (torch.Tensor): Bounding box coordinates (x, y, w, h) with
            center position (x, y), aspect ratio a, and height h.
            [bs, num_state, 1]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the mean vector (8 dimensional) and
                covariance matrix (8x8 dimensional) of the new track.
                Unobserved velocities are initialized to 0 mean.
                correspond to   [x, x', y, y', w/h, w/h', h, h']
        """
        N, dim_state, _ = state.shape
        new_state = torch.cat([state], dim=-1).reshape(N, self.dim_state,
                                                       1)  # [x, y, y', w,  h]

        P = torch.tensor([
            2 * self._std_weight_position,
            2 * self._std_weight_position,
            2 * self._std_weight_position,
            2 * self._std_weight_position,
        ]).to(state.device)
        P = torch.diag(P**2)
        w = state[:, 2, :]
        h = state[:, 3, :]  # [bs, 1]
        _dynamic = torch.cat([w, h, w, h], dim=1)  # noqa: E126
        _dynamic_2 = torch.square(torch.diag_embed(_dynamic))
        self.P = P * _dynamic_2
        self._reset_noise()
        self.Q = self._update_Q(new_state)
        self.R = self._update_R(new_state)
        return new_state, self.P

    def _update_Q(self, state: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            state (torch.Tensor): [bs, [x, y, w, h],1]

        Returns:
            torch.Tensor: The 8x8 dimensional covariance matrix.
        """
        self._reset_noise()
        batch_size, dim_state, _ = state.shape
        w = state[:, 2, :]
        h = state[:, 3, :]
        _dynamic = torch.cat([
            w,
            h,
            w,
            h,
        ], dim=1)  # noqa: E126
        if self.Q.ndim < 3:
            self.Q.unsqueeze_(0)
        self.Q = self.Q.to(state.device) * torch.square(
            torch.diag_embed(_dynamic))

        return self.Q

    def _update_R(self, state: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            state (torch.Tensor): [bs, [x,  y, w, , h, ],1]

        Returns:
            torch.Tensor: _description_
        """
        self._reset_noise()
        batch_size, dim_state, _ = state.shape
        w = state[:, 2, :]
        h = state[:, 3, :]
        _dynamic = torch.cat([w, h, w, h], dim=1)
        if self.R.ndim < 3:
            self.R.unsqueeze_(0)
        self.R = self.R.to(state.device) * torch.square(
            torch.diag_embed(_dynamic))
        return self.R

    def _reset_noise(self):
        Q = torch.tensor([
            self._std_weight_position,
            self._std_weight_position,
            self._std_weight_position,
            self._std_weight_position,
        ])
        self.Q = torch.square(torch.diag(Q))

        R = torch.tensor([
            self._std_weight_position, self._std_weight_position,
            self._std_weight_position, self._std_weight_position
        ])

        self.R = torch.diag(R**2)


class STMOTXYAHParams(MOTChallengeParams):

    def __init__(self,
                 dim_state: int = 4,
                 obs_ind: List[int] | None = [0, 1, 2, 3],
                 dt: int | float = 1,
                 std_weight_position: float = 1. / 20,
                 update_noise: bool = False):
        super().__init__(dim_state=dim_state,
                         obs_ind=obs_ind,
                         dt=dt,
                         std_weight_position=std_weight_position,
                         update_noise=update_noise)
        # corresponds to [x, x', x'', y, y', y'', w, w', w'', h, h', h'',]

        F = torch.tensor([[1.]])
        self.F = torch.kron(torch.eye(self.dim_state), F)
        H = torch.eye(dim_state)
        self.H = H[obs_ind]
        Q = torch.tensor([
            self._std_weight_position,
            self._std_weight_position,
            1e-2,
            self._std_weight_position,
        ])
        self.Q = torch.square(torch.diag(Q))

        R = torch.tensor([
            self._std_weight_position, self._std_weight_position, 1e-1,
            self._std_weight_position
        ])

        self.R = torch.diag(R**2)

    def initiate(self, state: torch.Tensor,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            state (torch.Tensor): Bounding box coordinates (x, y, a, h) with
            center position (x, y), aspect ratio a, and height h.
            [bs, num_state, 1]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the mean vector (8 dimensional) and
                covariance matrix (8x8 dimensional) of the new track.
                Unobserved velocities are initialized to 0 mean.
                correspond to   [x,  y, w/h, h]
        """
        N, dim_state, _ = state.shape
        new_state = torch.cat([state], dim=-1).reshape(
            N, self.dim_state, 1)  # [x, x', y, y', w/h, w/h', h, h']

        P = torch.tensor([
            2 * self._std_weight_position,
            2 * self._std_weight_position,
            1e-2,
            2 * self._std_weight_position,
        ]).to(state.device)
        P = torch.diag(P**2)
        _dynamic = state[:, 3, :]  # [bs, 1]
        _dynamic = torch.cat([
            _dynamic,
            _dynamic,
            torch.ones_like(_dynamic),
            _dynamic,
        ],
                             dim=1).to(state.device)  # noqa: E126
        _dynamic_2 = torch.square(torch.diag_embed(_dynamic))
        self.P = P * _dynamic_2
        self._reset_noise()
        self.Q = self._update_Q(new_state)
        self.R = self._update_R(new_state)
        return new_state, self.P

    def _update_Q(self, state: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            state (torch.Tensor): [bs, [x, y, w, h],1]

        Returns:
            torch.Tensor: The 8x8 dimensional covariance matrix.
        """
        self._reset_noise()
        batch_size, dim_state, _ = state.shape
        w = state[:, 2, :]
        h = state[:, 3, :]
        _dynamic = torch.cat([
            w,
            h,
            w,
            h,
        ], dim=1)  # noqa: E126
        if self.Q.ndim < 3:
            self.Q.unsqueeze_(0)
        self.Q = self.Q.to(state.device) * torch.square(
            torch.diag_embed(_dynamic))

        return self.Q

    def _update_R(self, state: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            state (torch.Tensor): [bs, [x,  y, w, , h, ],1]

        Returns:
            torch.Tensor: _description_
        """
        self._reset_noise()
        batch_size, dim_state, _ = state.shape
        w = state[:, 2, :]
        h = state[:, 3, :]
        _dynamic = torch.cat([w, h, w, h], dim=1)
        if self.R.ndim < 3:
            self.R.unsqueeze_(0)
        self.R = self.R.to(state.device) * torch.square(
            torch.diag_embed(_dynamic))
        return self.R

    def _reset_noise(self):
        Q = torch.tensor([
            self._std_weight_position,
            self._std_weight_position,
            1e-2,
            self._std_weight_position,
        ])
        self.Q = torch.square(torch.diag(Q))

        R = torch.tensor([
            self._std_weight_position, self._std_weight_position, 1e-1,
            self._std_weight_position
        ])

        self.R = torch.diag(R**2)
