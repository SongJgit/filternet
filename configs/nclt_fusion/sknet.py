from mmengine.config import read_base
from filternet.models import NCLTFusionSplitKalmanNet
with read_base():
    from .base import DATA, params, METRIC, TRAINER, MONITOR, epoch, project

slide_win_size = 8
detach_step = 2
lr = 0.001
wd = 1e-4
train_seq_len = 100

is_unsupervised = False
LOGGER = dict(project=project, name=f'SKNet_unsup{is_unsupervised}', offline=False)

OPTIMIZER = dict(type='Adam', params=dict(lr=lr, weight_decay=wd))
# SCHEDULER = dict(type='CommonScheduler',
#                  scheduler=dict(type='CosineAnnealingLR',
#                                 params=dict(T_max=epoch // 1)))
LOSS = dict(
    loss_name='SmoothL1Loss',
    params=dict(reduction='mean'),
    pred_loss_mask=[True, True, False, False, False, False],
    target_loss_mask=[True, True],  # must match state
)

TRAINER.update(slide_win_size=slide_win_size, detach_step=detach_step)
DATA.train_dataset.seq_len = train_seq_len

MODEL = dict(
    type=NCLTFusionSplitKalmanNet,
    params=params,
    loss_cfg=LOSS,
    is_unsupervised=is_unsupervised,
    gru_scale_s=2,
    nGRU=2,
)
