from mmengine.config import read_base
from filternet.models import KalmanNetArch2
with read_base():
    from .base import lorenz_params, DATA, METRIC, TRAINER, MONITOR, epoch, project

lr = 0.001
wd = 1e-3
is_unsupervised = False
slide_win_size = 4
detach_step = 2
LOGGER = dict(project=project, name=f'KNetArch2_is_unsupervised{is_unsupervised}', offline=False)

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
    type=KalmanNetArch2,
    params=lorenz_params,
    in_mult_KNet=5,
    out_mult_KNet=40,
    loss_cfg=LOSS,
    is_unsupervised=is_unsupervised,
)
