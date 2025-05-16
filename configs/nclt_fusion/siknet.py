from mmengine.config import read_base
from filternet.models.hybrid_model.kalman_net.fusion_siknet import NCLTFusionSemanticIndependentKalmanNet
with read_base():
    from .base import DATA, params, METRIC, TRAINER, MONITOR, epoch, project

slide_win_size = None
detach_step = None
lr = 0.005
wd = 0
train_seq_len = 50

is_unsupervised = False
LOGGER = dict(project=project, name=f'SIKNet_unsup{is_unsupervised}', offline=False)

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

TRAINER.update(slide_win_size=slide_win_size, detach_step=detach_step, epochs=epoch)
DATA.train_dataset.seq_len = train_seq_len

MODEL = dict(type=NCLTFusionSemanticIndependentKalmanNet,
             params=params,
             loss_cfg=LOSS,
             is_unsupervised=is_unsupervised,
             psqr_emb_channels=[4, 4, 4, 4])
