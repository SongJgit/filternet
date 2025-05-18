from mmengine.config import read_base
from filternet.models.hybrid_model import SplitKalmanNet
with read_base():
    from .base import lorenz_params, DATA, METRIC, TRAINER, MONITOR, epoch, project

lr = 0.001
wd = 1e-3
is_unsupervised = False

LOGGER = dict(project=project, name=f'SKNet_is_unsupervised{is_unsupervised}', offline=False)
OPTIMIZER = dict(type='Adam', params=dict(lr=lr, weight_decay=wd))
SCHEDULER = dict(type='CommonScheduler', scheduler=dict(type='CosineAnnealingLR', params=dict(T_max=epoch // 1)))

LOSS = dict(
    loss_name='SmoothL1Loss',
    params=dict(reduction='mean'),
    pred_loss_mask=None,
    target_loss_mask=None,
)

MODEL = dict(
    type=SplitKalmanNet,
    params=lorenz_params,
    loss_cfg=LOSS,
    is_unsupervised=is_unsupervised,
)
