from mmengine.config import read_base
from filternet.models import NCLTFusionKalmanNetArch1
with read_base():
    from .base import DATA, params, METRIC, TRAINER, MONITOR, epoch, project

slide_win_size = None
detach_step = 2
lr = 1e-3
wd = 1e-4

is_unsupervised = False
LOGGER = dict(project=project, name=f'KNetArch1_unsup{is_unsupervised}', offline=False)

OPTIMIZER = dict(type='Adam', params=dict(lr=lr, weight_decay=wd))
# SCHEDULER = dict(type='CommonScheduler',
#                  scheduler=dict(type='CosineAnnealingLR',
#                                 params=dict(T_max=epoch // 1)))
LOSS = dict(
    loss_name='MSELoss',
    params=dict(reduction='mean'),
    pred_loss_mask=[True, True, False, False, False, False],
    target_loss_mask=[True, True],  # must match state
)

TRAINER.update(slide_win_size=slide_win_size, detach_step=detach_step)
MODEL = dict(type=NCLTFusionKalmanNetArch1, params=params, c3=False, loss_cfg=LOSS, is_unsupervised=is_unsupervised)
