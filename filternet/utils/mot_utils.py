import torch
from .logger import logger
from typing import List, Callable, Dict, Any, Tuple
from .bbox_mode import bbox_cxcyah_to_xyxy, bbox_x1y1wh_to_xyxy, bbox_cxcywh_to_xyxy
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision as MAP
from torchmetrics.detection import IntersectionOverUnion as IOU
from .metrics import MSE
import numpy as np
from enum import Enum


class MOTClassesID(Enum):
    MOT = [
        dict(id=1, name='pedestrian'),  # only pedestrian
        dict(id=2, name='person_on_vehicle'),
        dict(id=3, name='car'),
        dict(id=4, name='bicycle'),
        dict(id=5, name='motorbike'),
        dict(id=6, name='non_mot_vehicle'),
        dict(id=7, name='static_person'),
        dict(id=8, name='distractor'),
        dict(id=9, name='occluder'),
        dict(id=10, name='occluder_on_ground'),
        dict(id=11, name='occluder_full'),
        dict(id=12, name='reflection'),
        dict(id=13, name='crowd'),
    ]
    SNMOT = [
        dict(id=-1, name='soccer_baller'),
    ]
    DanceTrack = [
        dict(id=1, name='dancer'),
    ]
    CIAAE_MOT24 = [
        dict(id=0, name='airplane'),
    ]
    CIAAE_SOT24 = [
        dict(id=0, name='airplane'),
    ]
    Infrared_MOT = [
        dict(id=0, name='airplane'),
    ]
    Helicopter_SOT = [
        dict(id=0, name='helicopter'),
    ]

    @classmethod
    def classes2id(cls, name):
        CLASSES = cls.get_classes(name)
        return {c['name']: c['id'] for c in CLASSES}

    @classmethod
    def id2classes(cls,
                   name,
                   classes: List[str] | None = None) -> Dict[int, str]:
        CLASSES2ID = cls.classes2id(name)
        if classes is None:
            classes = CLASSES2ID
        elif len(classes) == 0:
            classes = list(CLASSES2ID.keys()
                           )  # return all classes if classes is empty or None
        elif not isinstance(classes, list):
            classes = [classes]
        try:
            ID2CLASSES = {CLASSES2ID[cls]: cls
                          for cls in classes
                          }  # {1: 'pedestrian', 2: 'person_on_vehicle', ...}
        except Exception as e:
            raise ValueError(
                f'classes not in the dataset, available classes are {list(CLASSES2ID.keys())}, {e}'
            )
        return ID2CLASSES

    @classmethod
    def get_classes(cls, dataset_name) -> List[Dict[str, Any]]:
        if hasattr(cls, dataset_name):
            CLASSES = cls[dataset_name].value
        else:
            raise ValueError(f'Unsupported datasets_name: {dataset_name},',
                             f'Must be one of {list(cls.__members__.keys())}')
        return CLASSES


def inverse_xyah_bbox(bboxes: torch.Tensor, height, width) -> torch.Tensor:
    """_summary_

    Args:
        bboxes (torch.Tensor): [[x,y,a,h], len],
        height (_type_): _description_
        width (_type_): _description_

    Returns:
        _type_: _description_
    """

    norm_data = torch.tensor([width, height, 1,
                              height]).reshape(-1, 1).to(bboxes.device)
    return bboxes * norm_data


def inverse_xywh_bbox(bboxes: torch.Tensor, height, width) -> torch.Tensor:
    """_summary_

    Args:
        bboxes (torch.Tensor): [[x,y,a,h], len],
        height (_type_): _description_
        width (_type_): _description_

    Returns:
        _type_: _description_
    """
    norm_data = torch.tensor([width, height, width,
                              height]).reshape(-1, 1).to(bboxes.device)
    return bboxes * norm_data


def collect_mot_results(res, batch, bbox_mode, transforms, pred_metric_mask,
                        tgt_metric_mask) -> torch.Tensor:
    """for filter.results.

    Args:
        res (_type_): _description_
        batch (_type_): _description_
        bbox_mode (_type_): _description_
        transforms (_type_): _description_
        pred_metric_mask (_type_): _description_
        tgt_metric_mask (_type_): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    valid_step_mask = batch['valid_step_mask']  # [bs, step]
    frame_ids = torch.stack([data['frame_ids'] for data in batch['data_info']])

    observations = batch['inputs'].detach().cpu()  # [bs, 4, seq_len]
    origin_sequences = torch.stack(
        [data['origin_sequence'] for data in batch['data_info']])

    origin_sequence = origin_sequences[:, 1:5, :]
    visibility = origin_sequences[:, -1, :]
    video_name = [data['name'] for data in batch['data_info']]
    hw = [[data['height'], data['width']] for data in batch['data_info']]
    track_indices = [data['global_track_id'] for data in batch['data_info']]
    visibility = visibility != 0

    preds = []
    obs = []
    tgts = []

    if bbox_mode == 'cxcyah':
        inverse_func = inverse_xyah_bbox
        convert_func = bbox_cxcyah_to_xyxy
    elif bbox_mode == 'x1y1wh':
        inverse_func = inverse_xywh_bbox
        convert_func = bbox_x1y1wh_to_xyxy
    elif bbox_mode == 'cxcywh':
        inverse_func = inverse_xywh_bbox
        convert_func = bbox_cxcywh_to_xyxy
    else:
        raise NotImplementedError

    for idx in range(res.shape[0]):  # batch_size
        step_mask = valid_step_mask[idx] & visibility[idx].bool()
        valid_frame_id = torch.masked_select(frame_ids[idx], step_mask)
        valid_origin = torch.masked_select(origin_sequence[idx].cpu(),
                                           step_mask[None].cpu()).reshape(
                                               origin_sequence[idx].shape[0],
                                               -1)
        valid_res = torch.masked_select(res[idx].cpu(),
                                        step_mask[None].cpu()).reshape(
                                            res[idx].shape[0], -1)
        valid_ob = torch.masked_select(observations[idx],
                                       step_mask[None].cpu()).reshape(
                                           observations[idx].shape[0], -1)

        height = hw[idx][0]
        width = hw[idx][1]

        valid_res = valid_res[pred_metric_mask, :]
        valid_origin = valid_origin[tgt_metric_mask, :]

        valid_ob = valid_ob[tgt_metric_mask, :]

        if transforms:
            valid_ob = inverse_func(valid_ob, height=height, width=width)
            valid_res = inverse_func(valid_res, height=height, width=width)
            valid_origin = inverse_func(valid_origin,
                                        height=height,
                                        width=width)

        valid_ob_xyxy = convert_func(valid_ob.T)  # [seq_len, 4]
        valid_res_xyxy = convert_func(valid_res.T)
        valid_origin_xyxy = convert_func(valid_origin.T)

        valid_res_mask = ~torch.isnan(valid_res_xyxy).any(dim=1).flatten()
        valid_ob_xyxy = valid_ob_xyxy[valid_res_mask, :]
        valid_res_xyxy = valid_res_xyxy[valid_res_mask, :]
        valid_origin_xyxy = valid_origin_xyxy[valid_res_mask, :]

        valid_frame_id = valid_frame_id[valid_res_mask]
        track_preds = []
        track_tgts = []
        track_obs = []
        track_id = track_indices[idx]
        for i in range(valid_res_xyxy.shape[0]):  # [n, 4]
            # print(valid_target[[i]],valid_origin[[i]])
            track_preds.append(
                dict(
                    boxes=valid_res_xyxy[[i]],
                    # dict(boxes=valid_ob_xyxy[[i]], # for observation error.
                    scores=torch.tensor([1.]),
                    labels=torch.tensor([0]),
                    frame_id=valid_frame_id[[i]].to(torch.int),
                    track_id=track_id,
                    video_name=video_name[idx]))
            track_tgts.append(
                dict(
                    boxes=valid_origin_xyxy[[i]],
                    labels=torch.tensor([0]),
                    frame_id=valid_frame_id[[i]].to(torch.int),
                    track_id=track_id,
                    aspect_ratio=(valid_origin_xyxy[[i]][:, 2] -
                                  valid_origin_xyxy[[i]][:, 0]) /
                    (valid_origin_xyxy[[i]][:, 3] -
                     valid_origin_xyxy[[i]][:, 1]),  # w/h
                    video_name=video_name[idx]))

            track_obs.append(
                dict(boxes=valid_ob_xyxy[[i]],
                     labels=torch.tensor([0]),
                     frame_id=valid_frame_id[[i]].to(torch.int),
                     track_id=track_id,
                     video_name=video_name[idx]))
        if len(track_preds) == 0:
            continue
        preds.append(track_preds)
        tgts.append(track_tgts)
        obs.append(track_obs)

    return preds, tgts, obs


def _compute_metric(preds: List[List | Dict], tgts: List[List | Dict],
                    obs: List[List | Dict]) -> Dict:

    if isinstance(preds[0], dict):
        preds = [preds]
    elif not isinstance(preds[0], list):
        raise NotImplementedError
    rmse_fn = MSE(squared=False)
    map_metric = MAP()
    iou_metric = IOU()
    step_iou_metric = IOU()
    obs_step_iou_metric = IOU()  # [dict(video_name, ious =[...])]

    all_track_step_iou = []  # [dict(video_name, ious =[...])]

    for track_preds, track_tgts, track_obs in zip(preds, tgts, obs):
        map_metric.update(track_preds, track_tgts)
        iou_metric.update(track_preds, track_tgts)
        ious = []
        obs_ious = []
        frame_ids = []
        aspect_ratios = []
        for valid_res, valid_origin, valid_obs in zip(track_preds, track_tgts,
                                                      track_obs):
            valid_res_xyxy = valid_res['boxes']
            valid_origin_xyxy = valid_origin['boxes']
            rmse_fn.update(valid_res_xyxy.flatten(),
                           valid_origin_xyxy.flatten())
            step_iou_metric.update([valid_res], [valid_origin])
            step_iou = step_iou_metric.compute()
            step_iou_metric.reset()
            ious.append(step_iou['iou'].item())
            frame_ids.append(valid_res['frame_id'].item())

            obs_step_iou_metric.update([valid_obs], [valid_origin])
            obs_step_iou = obs_step_iou_metric.compute()
            obs_step_iou_metric.reset()
            obs_ious.append(obs_step_iou['iou'].item())

            aspect_ratios.append(valid_origin['aspect_ratio'].item())

        track_step_iou = dict(video_name=track_preds[0]['video_name'],
                              frame_id=np.array(frame_ids),
                              track_id=track_preds[0]['track_id'],
                              ious=np.array(ious),
                              obs_ious=np.array(obs_ious),
                              aspect_ratios=aspect_ratios)
        all_track_step_iou.append(track_step_iou)
    rmse = rmse_fn.compute()
    rmse_fn.reset()
    map = map_metric.compute()
    iou = iou_metric.compute()
    metrics = dict(iou=iou, map=map, rmse=rmse, step_iou=all_track_step_iou)
    return metrics


def run_mot_filter(
        loader, model: Callable, pred_metric_mask: List[bool] | Tensor,
        tgt_metric_mask: List[bool] | Tensor) -> Tuple[List, List, List, Dict]:
    """_summary_

    Args:
        loader (_type_): _description_
        model (Callable): _description_
        pred_metric_mask (List[bool] | Tensor): _description_
        tgt_metric_mask (List[bool] | Tensor): _description_

    Returns:
        Tuple[List, List, List, Dict]: _description_
    """
    # loader = data_module.test_dataloader()
    # run filter
    preds = []
    obs = []
    tgts = []

    bbox_mode = loader.dataset.state_mode
    for batch in loader:
        model.init_beliefs(batch['initial_state'])
        # print(batch['initial_state'])

        res = model.forward_loop(batch['inputs']).detach().cpu()
        p, t, o = collect_mot_results(res, batch, bbox_mode,
                                      loader.dataset.transforms,
                                      pred_metric_mask, tgt_metric_mask)
        # bs, dim_state, _ = res.shape
        preds.extend(p)
        tgts.extend(t)
        obs.extend(o)

    metrics = _compute_metric(preds, tgts, obs)
    logger.info(f"{bbox_mode.upper()} Prediction Error: {metrics['iou']},\n"
                f"{metrics['map']}")
    logger.info(f"RMSE:{metrics['rmse']}")
    return preds, tgts, obs, metrics


def get_mot_metric(predictions, bbox_mode, transforms, pred_metric_mask,
                   tgt_metric_mask) -> Tuple[List, List, List, Dict]:
    """for Trainer.predict, like collect_mot_results.

    Args:
        predictions (_type_): _description_
        bbox_mode (_type_): _description_
        transforms (_type_): _description_
        pred_metric_mask (_type_): _description_
        tgt_metric_mask (_type_): _description_

    Returns:
        Tuple[List, List, List, Dict]: _description_
    """

    preds = []
    obs = []
    tgts = []

    for batches in predictions:
        output = batches['preds']
        batch = batches['batch']
        p, t, o = collect_mot_results(output, batch, bbox_mode, transforms,
                                      pred_metric_mask, tgt_metric_mask)
        preds.extend(p)
        tgts.extend(t)
        obs.extend(o)

    metrics = _compute_metric(preds, tgts, obs)
    logger.info(f"{bbox_mode.upper()} Prediction Error: {metrics['iou']},\n"
                f"{metrics['map']}")
    logger.info(f"RMSE:{metrics['rmse']}")

    return preds, tgts, obs, metrics


def collect_mot_results_for_loss(res, batch, transforms, pred_loss_mask,
                                 tgt_loss_mask) -> Dict[str, Tensor]:
    valid_step_mask = batch['valid_step_mask']
    origin_sequence = batch['targets']
    # observations = batch['inputs'].detach()  # [bs, 4, seq_len]
    # origin_sequence = origin_sequences[:, 1:5, :]
    # visibility = origin_sequences[:, -1, :]
    # video_name = [data['name'] for data in batch['data_info']]
    hw = [[data['height'], data['width']] for data in batch['data_info']]
    visibility = torch.ones_like(valid_step_mask)
    visibility = visibility != 0

    preds = []
    tgts = []

    bbox_mode = batch['data_info'][0]['box_mode']

    if bbox_mode == 'cxcyah':
        inverse_func = inverse_xyah_bbox
        convert_func = bbox_cxcyah_to_xyxy
    elif bbox_mode == 'x1y1wh':
        inverse_func = inverse_xywh_bbox
        convert_func = bbox_x1y1wh_to_xyxy
    elif bbox_mode == 'cxcywh':
        inverse_func = inverse_xywh_bbox
        convert_func = bbox_cxcywh_to_xyxy
    else:
        raise NotImplementedError

    for idx in range(res.shape[0]):
        step_mask = valid_step_mask[idx] & visibility[idx].bool()
        valid_origin = torch.masked_select(origin_sequence[idx],
                                           step_mask[None]).reshape(
                                               origin_sequence[idx].shape[0],
                                               -1)
        valid_res = torch.masked_select(res[idx], step_mask[None]).reshape(
            res[idx].shape[0], -1)
        # valid_ob = torch.masked_select(observations[idx],
        #                                step_mask[None]).reshape(
        #                                    observations[idx].shape[0], -1)

        height = hw[idx][0]
        width = hw[idx][1]
        valid_res = valid_res[pred_loss_mask, :]
        valid_origin = valid_origin[tgt_loss_mask, :]
        if transforms:
            valid_res = inverse_func(valid_res, height=height, width=width)
            valid_origin = inverse_func(valid_origin,
                                        height=height,
                                        width=width)

        valid_res_xyxy = convert_func(valid_res.T)
        valid_origin_xyxy = convert_func(valid_origin.T)

        valid_res_mask = ~torch.isnan(valid_res_xyxy).any(dim=1).flatten()
        valid_res_xyxy = valid_res_xyxy[valid_res_mask, :]
        valid_origin_xyxy = valid_origin_xyxy[valid_res_mask, :]

        preds.append(valid_res_xyxy)
        tgts.append(valid_origin_xyxy)

        # (n,)
    return dict(preds=torch.cat(preds), tgts=torch.cat(tgts))
