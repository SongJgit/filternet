import torch
from typing import Union
from torch import Tensor
from numpy import ndarray


def bbox_cxcyah_to_xyxy(bboxes) -> Union[Tensor, ndarray]:
    """Convert bbox coordinates from (cx, cy, ratio, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    tensor2numpy = False
    if not isinstance(bboxes, torch.Tensor):
        tensor2numpy = True
        bboxes = torch.from_numpy(bboxes)
    cx, cy, ratio, h = bboxes.split((1, 1, 1, 1), dim=-1)
    w = ratio * h
    x1y1x2y2 = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
    x1y1x2y2 = torch.cat(x1y1x2y2, dim=-1)
    if tensor2numpy:
        x1y1x2y2 = x1y1x2y2.numpy()
    return x1y1x2y2


def bbox_cxcywh_to_xyxy(bboxes) -> Union[Tensor, ndarray]:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    tensor2numpy = False
    if not isinstance(bboxes, torch.Tensor):
        tensor2numpy = True
        bboxes = torch.from_numpy(bboxes)
    cx, cy, w, h = bboxes.split((1, 1, 1, 1), dim=-1)
    x1y1x2y2 = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
    x1y1x2y2 = torch.cat(x1y1x2y2, dim=-1)
    if tensor2numpy:
        x1y1x2y2 = x1y1x2y2.numpy()
    return x1y1x2y2


def bbox_xyxy_to_x1y1wh(bbox) -> Union[Tensor, ndarray]:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (x1, y1, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) or (4, ) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    tensor2numpy = False
    if not isinstance(bbox, torch.Tensor):
        tensor2numpy = True
        bbox = torch.from_numpy(bbox)
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [x1, y1, (x2 - x1), (y2 - y1)]
    bbox_new = torch.cat(bbox_new, dim=-1)
    if tensor2numpy:
        bbox_new = bbox_new.numpy()
    return bbox_new


def bbox_cxcywh_to_x1y1wh(bbox) -> Union[Tensor, ndarray]:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) or (4, ) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    tensor2numpy = False
    if not isinstance(bbox, torch.Tensor):
        tensor2numpy = True
        bbox = torch.from_numpy(bbox)
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), w, h]
    bbox_new = torch.cat(bbox_new, dim=-1)
    if tensor2numpy:
        bbox_new = bbox_new.numpy()
    return bbox_new


def bbox_xyxy_to_cxcyah(bboxes) -> Union[Tensor, ndarray]:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, ratio, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    tensor2numpy = False
    if not isinstance(bboxes, torch.Tensor):
        tensor2numpy = True
        bboxes = torch.from_numpy(bboxes)
    cx = (bboxes[:, 2] + bboxes[:, 0]) / 2
    cy = (bboxes[:, 3] + bboxes[:, 1]) / 2
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    xyah = torch.stack([cx, cy, w / h, h], -1)
    if tensor2numpy:
        xyah = xyah.numpy()
    return xyah


def bbox_x1y1wh_to_cxcyah(bboxes) -> Union[Tensor, ndarray]:
    """Convert bbox coordinates from (x1, y1, w, h) to (cx, cy, ratio, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    tensor2numpy = False
    if not isinstance(bboxes, torch.Tensor):
        tensor2numpy = True
        bboxes = torch.from_numpy(bboxes)

    cx = (bboxes[:, 0] + bboxes[:, 2] + bboxes[:, 0]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3] + bboxes[:, 1]) / 2
    w = bboxes[:, 2]
    h = bboxes[:, 3]
    xyah = torch.stack([cx, cy, w / h, h], -1)
    if tensor2numpy:
        xyah = xyah.numpy()
    return xyah


def bbox_x1y1wh_to_cxcywh(bboxes) -> Union[Tensor, ndarray]:
    """Convert bbox coordinates from (x1, y1, w, h) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    tensor2numpy = False
    if not isinstance(bboxes, torch.Tensor):
        tensor2numpy = True
        bboxes = torch.from_numpy(bboxes)

    cx = (bboxes[:, 0] + bboxes[:, 2] + bboxes[:, 0]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3] + bboxes[:, 1]) / 2
    w = bboxes[:, 2]
    h = bboxes[:, 3]
    xywh = torch.stack([cx, cy, w, h], -1)
    if tensor2numpy:
        xywh = xywh.numpy()
    return xywh


def bbox_x1y1wh_to_xyxy(bboxes) -> Union[Tensor, ndarray]:
    """Convert bbox coordinates from (x, y, w, h) to (x, y, x, y).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    tensor2numpy = False
    if not isinstance(bboxes, torch.Tensor):
        tensor2numpy = True
        bboxes = torch.from_numpy(bboxes)

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 0] + bboxes[:, 2]
    y2 = bboxes[:, 1] + bboxes[:, 3]
    xyah = torch.stack([x1, y1, x2, y2], -1)
    if tensor2numpy:
        xyah = xyah.numpy()
    return xyah


def bbox_cxcyah_to_x1y1wh(bboxes) -> Union[Tensor, ndarray]:
    """Convert bbox coordinates from (cx, cy, ratio, h) to (x1, y1, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    tensor2numpy = False
    if not isinstance(bboxes, torch.Tensor):
        tensor2numpy = True
        bboxes = torch.from_numpy(bboxes)
    cx, cy, ratio, h = bboxes.split((1, 1, 1, 1), dim=-1)
    w = ratio * h
    x1y1wh = torch.cat([cx - w / 2.0, cy - h / 2.0, w, h], dim=-1)
    if tensor2numpy:
        x1y1wh = x1y1wh.numpy()
    return x1y1wh
