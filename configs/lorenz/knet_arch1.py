from mmengine.config import read_base
from filternet.models import KalmanNetArch1
with read_base():
    from .base import lorenz_params, DATA, METRIC, TRAINER, MONITOR, epoch, project

lr = 0.001
wd = 1e-3
slide_win_size = 4
detach_step = 2
is_unsupervised = False
LOGGER = dict(project=project, name=f'KNetArch1_is_unsupervised{is_unsupervised}', offline=False)

OPTIMIZER = dict(type='Adam', params=dict(lr=lr, weight_decay=wd))
SCHEDULER = dict(type='CommonScheduler', scheduler=dict(type='CosineAnnealingLR', params=dict(T_max=epoch // 1)))
TRAINER.update(slide_win_size=slide_win_size, detach_step=detach_step)
LOSS = dict(
    loss_name='MSELoss',
    params=dict(reduction='mean', loss_weight=1),
    pred_loss_mask=None,
    target_loss_mask=None,
)

MODEL = dict(
    type=KalmanNetArch1,
    params=lorenz_params,
    c3=False,  # if True, use F1, F3, F4
    loss_cfg=LOSS,
    is_unsupervised=is_unsupervised,
)
