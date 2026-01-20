from typing import Tuple, Union

import numpy as np
import torch

# cSpell: words arange
Tensor = Union[torch.Tensor, np.ndarray]


def cartesian2spherical(x: Tensor, y: Tensor, z: Tensor) -> Tuple[Tensor, ...]:
    """_summary_

    Args:
        x (np.array | torch.Tensor):
        y (np.array | torch.Tensor): _description_
        z (np.array | torch.Tensor): _description_

    Returns:
        _type_: range r, pitch theta and azimuth phi.
    """
    if isinstance(x, torch.Tensor):
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = torch.acos(z / r)
        phi = torch.atan2(y, x)
    else:
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
    return r, theta, phi


def spherical2cartesian(r: Tensor, theta: Tensor, phi: Tensor) -> Tuple[Tensor, ...]:
    """_summary_

    Args:
        r (np.array | torch.Tensor): range
        theta (np.array | torch.Tensor): pitch
        phi (np.array | torch.Tensor): azimuth

    Returns:
        Tuple[Tensor, Tensor, Tensor]: x,y,z
    """
    if isinstance(r, torch.Tensor):
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
    else:
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
    return x, y, z


def cartesian2polar(x: Tensor, y: Tensor) -> Tuple[Tensor, ...]:
    """_summary_

    Args:
        x (np.array | torch.Tensor): _description_
        y (np.array | torch.Tensor): _description_

    Returns:
        Tuple[Tensor, Tensor]: range r,  pitch theta
    """
    if isinstance(x, torch.Tensor):
        r = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x)
    else:
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
    return r, theta


def polar2cartesian(r: Tensor, theta: Tensor) -> Tuple[Tensor, ...]:
    """_summary_

    Args:
        r (np.array | torch.Tensor): range r
        theta (np.array | torch.Tensor): pitch theta

    Returns:
        Tuple[Tensor, Tensor]: x, y
    """
    if isinstance(r, torch.Tensor):
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
    else:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
    return x, y
