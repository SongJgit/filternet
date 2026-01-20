import torch
from .logger import logger
from typing import List, Callable, Dict
from .bbox_mode import bbox_cxcyah_to_xyxy, bbox_x1y1wh_to_xyxy, bbox_cxcywh_to_xyxy
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision as MAP
# from torchmetrics.detection import IntersectionOverUnion as IOU
from .metrics import MSE
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
        dict(id=13, name='crowd'), ]
    SNMOT = [
        dict(id=-1, name='soccer_baller'), ]
    DanceTrack = [
        dict(id=1, name='dancer'), ]
    CIAAE_MOT24 = [
        dict(id=0, name='airplane'), ]
    CIAAE_SOT24 = [
        dict(id=0, name='airplane'), ]
    Infrared_MOT = [
        dict(id=0, name='airplane'), ]
    Helicopter_SOT = [
        dict(id=0, name='helicopter'), ]
    VisDroneTrack = [
        dict(id=0, name='drone'), ]

    AntiUAV = [
        dict(id=1, name='drone'), ]

    SemiUAV = [
        dict(id=-1, name='drone'), ]

    @classmethod
    def classes2id(cls, name):
        CLASSES = cls.get_classes(name)
        return {c['name']: c['id'] for c in CLASSES}

    @classmethod
    def id2classes(cls, name, classes: List[str] | None = None):
        CLASSES2ID = cls.classes2id(name)
        if classes is None:
            classes = CLASSES2ID
        elif len(classes) == 0:
            classes = list(CLASSES2ID.keys())  # return all classes if classes is empty or None
        elif not isinstance(classes, list):
            classes = [classes]
        try:
            ID2CLASSES = {CLASSES2ID[cls]: cls for cls in classes}  # {1: 'pedestrian', 2: 'person_on_vehicle', ...}
        except Exception as e:
            raise ValueError(f'classes not in the dataset, available classes are {list(CLASSES2ID.keys())}, {e}')
        return ID2CLASSES

    @classmethod
    def get_classes(cls, dataset_name):
        if hasattr(cls, dataset_name):
            CLASSES = cls[dataset_name].value
        else:
            raise ValueError(f'Unsupported datasets_name: {dataset_name},',
                             f'Must be one of {list(cls.__members__.keys())}')
        return CLASSES


def inverse_xyah_bbox(bboxes: torch.Tensor, height: torch.Tensor | float, width: torch.Tensor | float):
    """_summary_

    Args:
        bboxes (torch.Tensor): [[x,y,a,h], len],
        height (_type_): _description_
        width (_type_): _description_

    Returns:
        _type_: _description_
    """
    if not torch.is_tensor(height):
        norm_data = torch.tensor([width, height, 1, height]).reshape(4, -1).to(bboxes.device)
    else:
        norm_data = torch.cat([width, height, torch.ones_like(height), height], dim=-1).reshape(4, -1).to(bboxes.device)
    if bboxes.shape[0] == 2:
        norm_data = norm_data[:2, :]
    return bboxes * norm_data


def inverse_xywh_bbox(bboxes: torch.Tensor, height: torch.Tensor | float, width: torch.Tensor | float):
    """_summary_

    Args:
        bboxes (torch.Tensor): [[x,y,w,h], len],
        height (_type_): _description_
        width (_type_): _description_

    Returns:
        _type_: _description_
    """
    if not torch.is_tensor(height):
        norm_data = torch.tensor([width, height, width, height]).reshape(4, -1).to(bboxes.device)
    else:
        norm_data = torch.cat([width, height, width, height], dim=-1).reshape(4, -1).to(bboxes.device)
    if bboxes.shape[0] == 2:
        norm_data = norm_data[:2, :]
    return bboxes * norm_data


def _compute_metric(preds: List[List | Dict], tgts: List[List | Dict], obs: List[List | Dict]):

    if isinstance(preds[0], dict):
        preds = [preds]
    elif not isinstance(preds[0], list):
        raise NotImplementedError
    rmse_fn = MSE(squared=False)
    map_metric = MAP(extended_summary=True, backend='faster_coco_eval')
    # iou_metric = IOU()
    # step_iou_metric = IOU()
    # obs_step_iou_metric = IOU()  # [dict(video_name, ious =[...])]

    all_track_step_iou = []  # [dict(video_name, ious =[...])]

    for track_preds, track_tgts, track_obs in zip(preds, tgts, obs):
        map_metric.update(track_preds, track_tgts)
        # iou_metric.update(track_preds, track_tgts)
        # ious = []
        # obs_ious = []
        frame_ids = []
        aspect_ratios = []
        for valid_res, valid_labels, valid_obs in zip(track_preds, track_tgts, track_obs):
            valid_res_xyxy = valid_res['boxes']
            valid_labels_xyxy = valid_labels['boxes']
            rmse_fn.update(valid_res_xyxy.flatten(), valid_labels_xyxy.flatten())
            # step_iou_metric.update([valid_res], [valid_labels])
            # step_iou = step_iou_metric.compute()
            # step_iou_metric.reset()
            # ious.append(step_iou['iou'].item())
            frame_ids.append(valid_res['frame_id'].item())

            # obs_step_iou_metric.update([valid_obs], [valid_labels])
            # obs_step_iou = obs_step_iou_metric.compute()
            # obs_step_iou_metric.reset()
            # obs_ious.append(obs_step_iou['iou'].item())

            aspect_ratios.append(valid_labels['aspect_ratio'].item())

        # track_step_iou = dict(video_name=track_preds[0]['video_name'],
        #                       frame_id=np.array(frame_ids),
        #                       track_id=track_preds[0]['track_id'],
        #                       ious=np.array(ious),
        #                       obs_ious=np.array(obs_ious),
        #                       aspect_ratios=aspect_ratios)
        # all_track_step_iou.append(track_step_iou)
    rmse = rmse_fn.compute()
    rmse_fn.reset()
    map = map_metric.compute()
    # iou = iou_metric.compute()
    iou = 0
    metrics = dict(iou=iou, map=map, rmse=rmse, step_iou=all_track_step_iou)
    return metrics


def collect_mot_results_for_loss(res: torch.tensor, batch: Dict, transforms: bool, pred_mask: List[bool] | Tensor,
                                 tgt_mask: List[bool] | Tensor):
    """_summary_
    Example:
        from torchmetrics.detection import IntersectionOverUnion as IOU
        iou = IOU()
        valid_step_mask = torch.ones((2,10), dtype= bool) # [batch_size, num_steps]
        targets = torch.randn(2, 4, 10) # # [batch_size, bbox, num_steps]
        inputs = torch.randn(2, 4, 10) # [batch_size, bbox, num_steps]
        batch= {
            'valid_step_mask': valid_step_mask,
            'targets': targets,
            'inputs': inputs,
            'data_info': [{'height': 100, 'width': 200,
                            'box_mode': 'cxcywh',
                            'frame_ids': torch.tensor([0,1,2,3,4,5,6,7,8,9]),
                            'name': 'test'}] * 2
        }
        res = torch.randn(2,4,10)

        out = collect_mot_results(res, batch, True, target_loss_mask, target_loss_mask)
        loss_fn = nn.SmoothL1Loss()
        iou(out['light_format']['preds'], out['light_format']['tgts']), loss_fn(out['preds'], out['tgts'])

    Args:
        res (torch.tensor): model output
        batch (Dict): _description_
        transforms (bool): _description_
        pred_mask (List[bool] | Tensor): _description_
        tgt_mask (List[bool] | Tensor): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    valid_step_mask = batch['valid_step_mask']  # [bs, step]
    frame_ids = torch.stack([data['frame_ids'] for data in batch['data_info']])

    observations = batch['inputs']  # [bs, 4, seq_len]
    labels = batch['targets']  # [bs, 4, seq_len]
    video_names_map = {i: data['name'] for i, data in enumerate(batch['data_info'])}
    hw = [[data['height'], data['width']] for data in batch['data_info']]
    hw = torch.tensor(hw)  # [bs, 2]

    bbox_mode = batch['data_info'][0]['box_mode']

    video_name_idx = torch.tensor(list(video_names_map.keys()), dtype=torch.long)

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

    waited_lbls = []
    waited_res = []
    waited_obs = []
    waited_frame_ids = []
    repeats = []
    # waited_video_names = []

    for idx in range(res.shape[0]):  # batch_size
        # each trajectory
        step_mask = valid_step_mask[idx]
        valid_frame_id = torch.masked_select(frame_ids[idx], step_mask)
        valid_labels = torch.masked_select(labels[idx], step_mask[None]).reshape(labels[idx].shape[0], -1)
        valid_res = torch.masked_select(res[idx], step_mask[None]).reshape(res[idx].shape[0], -1)
        valid_ob = torch.masked_select(observations[idx], step_mask[None]).reshape(observations[idx].shape[0], -1)

        waited_lbls.append(valid_labels)
        waited_res.append(valid_res)
        waited_obs.append(valid_ob)
        waited_frame_ids.append(valid_frame_id)
        # waited_video_names.append(video_name_idx[idx])

        repeats.append(valid_labels.shape[1])

    waited_lbls = torch.cat(waited_lbls, dim=1)  # [N, seq_len * bs]
    waited_res = torch.cat(waited_res, dim=1)
    waited_obs = torch.cat(waited_obs, dim=1)
    waited_frame_ids = torch.cat(waited_frame_ids, dim=0)
    hw = torch.repeat_interleave(hw, repeats=torch.tensor(repeats), dim=0)
    video_name_idx = torch.repeat_interleave(video_name_idx, repeats=torch.tensor(repeats), dim=0)

    valid_res = waited_res[pred_mask, :]
    valid_labels = waited_lbls[tgt_mask, :]
    valid_obs = waited_obs[tgt_mask, :]

    if transforms:
        valid_obs = inverse_func(valid_obs, height=hw[:, 0], width=hw[:, 1])
        valid_res = inverse_func(valid_res, height=hw[:, 0], width=hw[:, 1])
        valid_labels = inverse_func(valid_labels, height=hw[:, 0], width=hw[:, 1])

    valid_obs_xyxy = convert_func(valid_obs.T)  # [seq_len, 4]
    valid_res_xyxy = convert_func(valid_res.T)
    valid_labels_xyxy = convert_func(valid_labels.T)

    valid_res_mask = ~torch.isnan(valid_res_xyxy).any(dim=1).flatten()
    valid_obs_xyxy = valid_obs_xyxy[valid_res_mask, :]
    valid_res_xyxy = valid_res_xyxy[valid_res_mask, :]
    valid_labels_xyxy = valid_labels_xyxy[valid_res_mask, :]

    valid_frame_ids = waited_frame_ids[valid_res_mask.cpu()]
    valid_name_ids = video_name_idx[valid_res_mask.cpu()]

    return {
        'preds': valid_res_xyxy,
        'tgts': valid_labels_xyxy,
        'obs': valid_obs_xyxy,
        'valid_name_ids': valid_name_ids,
        'valid_frame_ids': valid_frame_ids,
        'video_names_map': video_names_map}


def collect_mot_results_for_metric(res: torch.tensor, batch: Dict, transforms: bool, pred_mask: List[bool] | Tensor,
                                   tgt_mask: List[bool] | Tensor):
    """_summary_
    Example:
        from torchmetrics.detection import IntersectionOverUnion as IOU
        iou = IOU()
        valid_step_mask = torch.ones((2,10), dtype= bool) # [batch_size, num_steps]
        targets = torch.randn(2, 4, 10) # # [batch_size, bbox, num_steps]
        inputs = torch.randn(2, 4, 10) # [batch_size, bbox, num_steps]
        batch= {
            'valid_step_mask': valid_step_mask,
            'targets': targets,
            'inputs': inputs,
            'data_info': [{'height': 100, 'width': 200,
                            'box_mode': 'cxcywh',
                            'frame_ids': torch.tensor([0,1,2,3,4,5,6,7,8,9]),
                            'name': 'test'}] * 2
        }
        res = torch.randn(2,4,10)

        out = collect_mot_results(res, batch, True, target_loss_mask, target_loss_mask)
        loss_fn = nn.SmoothL1Loss()
        iou(out['light_format']['preds'], out['light_format']['tgts']), loss_fn(out['preds'], out['tgts'])

    Args:
        res (torch.tensor): model output
        batch (Dict): _description_
        transforms (bool): _description_
        pred_mask (List[bool] | Tensor): _description_
        tgt_mask (List[bool] | Tensor): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    out = collect_mot_results_for_loss(res, batch, transforms, pred_mask, tgt_mask)

    valid_res_xyxy = out['preds']
    valid_labels_xyxy = out['tgts']
    valid_obs_xyxy = out['obs']
    valid_name_ids = out['valid_name_ids']
    valid_frame_ids = out['valid_frame_ids']
    video_names_map = {i: data['name'] for i, data in enumerate(batch['data_info'])}

    track_preds = []
    track_tgts = []
    track_obs = []

    for i in range(valid_res_xyxy.shape[0]):  # [n, 4]
        # each frame.
        # print(valid_target[[i]],valid_labels[[i]])
        video_name = video_names_map[int(valid_name_ids[i])]
        frame_id = valid_frame_ids[[i]].to(torch.int)

        res_box = valid_res_xyxy[[i]].cpu()
        label_box = valid_labels_xyxy[[i]].cpu()
        obs_box = valid_obs_xyxy[[i]].cpu()

        # 计算宽高比
        w = label_box[:, 2] - label_box[:, 0]
        h = label_box[:, 3] - label_box[:, 1]
        aspect_ratio = w / h

        # 构建结果字典
        track_preds.append(
            dict(
                boxes=res_box,
                # boxes = obs_box,
                scores=torch.tensor([1.]),
                labels=torch.tensor([0]),
                frame_id=frame_id,
                video_name=video_name))

        track_tgts.append(
            dict(boxes=label_box,
                 labels=torch.tensor([0]),
                 frame_id=frame_id,
                 aspect_ratio=aspect_ratio,
                 video_name=video_name))

        track_obs.append(dict(boxes=obs_box, labels=torch.tensor([0]), frame_id=frame_id, video_name=video_name))

    results = {'preds': track_preds, 'tgts': track_tgts, 'obs': track_obs}

    return results


def run_mot_filter(loader, model: Callable, pred_metric_mask: List[bool] | Tensor,
                   tgt_metric_mask: List[bool] | Tensor):
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
        out = collect_mot_results_for_metric(res, batch, bbox_mode, loader.dataset.transforms, pred_metric_mask,
                                             tgt_metric_mask)
        # bs, dim_state, _ = res.shape
        preds.extend(out['preds'])
        tgts.extend(out['tgts'])
        obs.extend(out['obs'])

    metrics = _compute_metric(preds, tgts, obs)
    logger.info(f"{bbox_mode.upper()} Prediction Error: {metrics['iou']},\n"
                f"AR: {metrics['map']['mar_1']}, AR@50: {metrics['map']['recall'][0,:,0, :].mean()}, \n"
                f"AR@75: {metrics['map']['recall'][5,:,0, :].mean()} ")
    logger.info(f"RMSE:{metrics['rmse']}")
    return preds, tgts, obs, metrics


def get_mot_metric(predictions, bbox_mode, transforms, pred_metric_mask, tgt_metric_mask):
    """for Trainer.predict, like collect_mot_results.

    Args:
        predictions (_type_): _description_
        bbox_mode (_type_): _description_
        transforms (_type_): _description_
        pred_metric_mask (_type_): _description_
        tgt_metric_mask (_type_): _description_

    Returns:
        _type_: _description_
    """
    preds = []
    obs = []
    tgts = []

    for batches in predictions:
        output = batches['preds']
        batch = batches['batch']
        out = collect_mot_results_for_metric(output, batch, bbox_mode, transforms, pred_metric_mask, tgt_metric_mask)

        preds.extend(out['preds'])
        tgts.extend(out['tgts'])
        obs.extend(out['obs'])

    metrics = _compute_metric(preds, tgts, obs)
    logger.info(f"{bbox_mode.upper()} Prediction Error: {metrics['iou']},\n"
                f"AR: {metrics['map']['mar_1']}, AR@50: {metrics['map']['recall'][0,:,0, :].mean()}, \n"
                f"AR@75: {metrics['map']['recall'][5,:,0, :].mean()} ")
    logger.info(f"RMSE:{metrics['rmse']}")

    return preds, tgts, obs, metrics
