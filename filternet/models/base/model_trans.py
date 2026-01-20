from typing import List

import torch

from filternet.registry import MODELTRANS

# Thanks https://note.yongcong.wang/Self-Driving/Prediction/imm-for-prediction/


class Trans:

    def __init__(self, n_axis: int = 1, turn_rate: bool = True) -> None:
        """CT Models with Unknown/Known Turn Rate.

        Args:
            dim (int, optional): _description_. Defaults to 1.
            1 means [x]
            2 means [x, y]
            3 means [x, y, z]
        """
        # Correspond to a = [x, x', x''].T,
        # CA -> CV T12.T @ a == [x, x'].T
        # For clear.
        if n_axis == 1:
            self.st_dim = n_axis * 1  # [x]
            self.cv_dim = n_axis * 2  # [x, x'].T
            self.ca_dim = n_axis * 3  # [x, x', x''].T

            if turn_rate:
                self.ct_dim = 3  # [x, x', omega].T
            else:
                self.ct_dim = 2
        elif n_axis == 2:
            self.st_dim = n_axis * 1  # [x, y]
            self.cv_dim = n_axis * 2  # [x, x', y, y'].T
            self.ca_dim = n_axis * 3  # [x, x', x'', y, y',y''].T
            if turn_rate:
                self.ct_dim = 5  # [x, x', y, y', omega].T
            else:
                self.ct_dim = 4

        elif n_axis == 3:
            self.st_dim = n_axis * 1  # [x, y, z]
            self.cv_dim = n_axis * 2  # [x, x', y, y', z, z'].T
            self.ca_dim = n_axis * 3  # [x, x', x'', y, y',y'', z, z', z''].T
            if turn_rate:
                self.ct_dim = 7  # [x, x', y, y', z, z', omega].T
            else:
                self.ct_dim = 6

        elif n_axis == 4:
            self.st_dim = n_axis * 1  # [x, y, z]
            self.cv_dim = n_axis * 2  # [x, x', y, y', w, w', h, h'].T
            self.ca_dim = n_axis * 3  # [x, x', x'', y, y', y'', w, w', w'', h, h', h''].T
            if turn_rate:
                self.ct_dim = 9  # [x, x', y, y', w, w', h, h', omega].T
            else:
                self.ct_dim = 8

        self.n_axis = n_axis
        T12 = torch.Tensor([[1, 0], [0, 1], [0, 0]])

        self.T12 = torch.kron(torch.eye(self.n_axis), T12)

        # a = [x, x', x''].T, CA -> CT self.T23@a == [x, x', 0].T
        # a =[x, x', omega].T, CT -> CA self.T23.T@a == [x, x', 0].T
        T23 = torch.Tensor([
            [1, 0, 0],
            [0, 1, 0], ])
        T23 = torch.kron(torch.eye(self.n_axis), T23)

        if turn_rate:
            self.T23 = torch.vstack([T23, torch.zeros(1, self.n_axis * 3)])
        else:
            self.T23 = T23
        # a = [x, x'], CV -> CT self.T13@a == [x, x',omega] if turn_rate.
        # a = [x, x',omega] CT -> CV self.T13.T@a == [x, x']
        self.T13 = self.T23 @ self.T12

        CV2ST = torch.Tensor([[1, 0]])
        self.CV2ST = torch.kron(torch.eye(self.n_axis), CV2ST)

        CA2ST = torch.Tensor([[1, 0, 0]])
        self.CA2ST = torch.kron(torch.eye(self.n_axis), CA2ST)

        self.model_trans: List[torch.Tensor] = []

    def to(self, device: str):
        for i in range(len(self.model_trans[0])):
            for j in range(len(self.model_trans[1])):
                self.model_trans[i][j] = self.model_trans[i][j].to(device)

    def __getitem__(self, key):
        return self.model_trans[key]

    def __len__(self):
        return len(self.model_trans)

    def __call__(self):
        """Return the model_trans matrix directly."""
        return self.model_trans


@MODELTRANS.register_module()
class SelfTrans(Trans):

    def __init__(self, n_axis: int = 1, mode: str = 'ca', num_models: int = 2) -> None:
        super().__init__(n_axis, turn_rate=False)
        """Self2Self or Self2Self
                   Self       Self
            Self Self2Self  Self2Self
            Self Self2Self  Self2Self
              .
              .
              .
        Args:
            n_axis (int, optional): _description_. Defaults to 1.
            1 means [x]
            2 means [x, y]
            3 means [x, y, z]
            mode (str, optional): 'ca' or 'cv'. Defaults to 'ca'.
        """
        if mode == 'ca':
            trans = torch.eye(self.ca_dim)
        elif mode == 'cv':
            trans = torch.eye(self.cv_dim)
        elif mode == 'ct':
            trans = torch.eye(self.ct_dim)
        else:
            print(f'{mode} is not supported')

        for i in range(num_models):
            _temp = []
            for j in range(num_models):
                _temp.append(trans)
            self.model_trans.append(_temp)


@MODELTRANS.register_module()
class CVATrans(Trans):

    def __init__(self, n_axis: int = 1) -> None:
        super().__init__(n_axis)
        """CV2CA or CA2CV
                CV     CA
            CV CV2CV  CA2CV
            CA CV2CA  CA2CA
        Args:
            n_axis (int, optional): _description_. Defaults to 1.
            1 means [x]
            2 means [x, y]
            3 means [x, y, z]
            4 means [x, y, w, h] in MOT.
        """
        # Correspond to a = [x, x', x''].T,
        # CA -> CV T12.T @ a == [x, x'].T

        self.model_trans = [[torch.eye(self.cv_dim), self.T12.T], [self.T12, torch.eye(self.ca_dim)]]


@MODELTRANS.register_module()
class CVTTrans(Trans):

    def __init__(self, n_axis: int = 1, turn_rate: bool = True):
        super().__init__(n_axis, turn_rate)
        """
            CV state/covariance to CT state/covariance, or CT2CV. only support x and y.

                CV      CT
            CV  CV2CV  CT2CV
            CT  CV2CT  CT2CT
        Args:
            n_axis (int, optional): _description_. Defaults to 1.
            1 means [x]
            2 means [x, y]
            3 means [x, y, z]
        """

        self.model_trans = [[torch.eye(self.cv_dim), self.T13.T], [self.T13, torch.eye(self.ct_dim)]]


@MODELTRANS.register_module()
class CATTrans(Trans):

    def __init__(self, n_axis: int = 1, turn_rate: bool = True):
        super().__init__(n_axis, turn_rate)
        """
            CV state/covariance to CT state/covariance, or CT2CV. only support x and y.

                CA     CT
            CA CA2CA  CT2CA
            CT CA2CT  CT2CT
        Args:
            n_axis (int, optional): _description_. Defaults to 1.
            1 means [x]
            2 means [x, y]
            3 means [x, y, z]
        """

        self.model_trans = [[torch.eye(self.ca_dim), self.T23.T], [self.T23, torch.eye(self.ct_dim)]]


@MODELTRANS.register_module()
class CVATTrans(Trans):

    def __init__(self, n_axis: int = 1, turn_rate: bool = True):
        super().__init__(n_axis, turn_rate)
        """
            CV2CA2CT.

                CV       CA     CT
            CV CV2CV   CA2CV   CT2CV
            CA CV2CA   CA2CA   CT2CA
            CT CV2CT   CA2CT   CT2CT
        Args:
            n_axis (int, optional): _description_. Defaults to 1.
            1 means [x]
            2 means [x, y]
            3 means [x, y, z]
        """

        self.model_trans = [[torch.eye(self.cv_dim), self.T12.T, self.T13.T],
                            [self.T12, torch.eye(self.ca_dim), self.T23.T],
                            [self.T13, self.T23, torch.eye(self.ct_dim)]]


@MODELTRANS.register_module()
class STCVTrans(Trans):

    def __init__(self, n_axis: int = 1, turn_rate: bool = True):
        super().__init__(n_axis, turn_rate)
        """
            static state/covariance to CV state/covariance, or CT2CV. only support x and y.

                ST      CV
            ST  ST2ST  CV2ST
            CV  ST2CV  CV2CV
        Args:
            n_axis (int, optional): _description_. Defaults to 1.
            1 means [x]
            2 means [x, y]
            3 means [x, y, z]
            4 means [x, y, w, h] or [x, y, a, h]
        """

        self.model_trans = [[torch.eye(self.st_dim), self.CV2ST], [self.CV2ST.T, torch.eye(self.cv_dim)]]


@MODELTRANS.register_module()
class STCATrans(Trans):

    def __init__(self, n_axis: int = 1, turn_rate: bool = True):
        super().__init__(n_axis, turn_rate)
        """
            static state/covariance to CV state/covariance, or CT2CV. only support x and y.

                ST      CA
            ST  ST2ST  CA2ST
            CA  ST2CA  CA2CA
        Args:
            n_axis (int, optional): _description_. Defaults to 1.
            1 means [x]
            2 means [x, y]
            3 means [x, y, z]
            4 means [x, y, w, h] or [x, y, a, h]
        """

        self.model_trans = [[torch.eye(self.st_dim), self.CA2ST], [self.CA2ST.T, torch.eye(self.ca_dim)]]
