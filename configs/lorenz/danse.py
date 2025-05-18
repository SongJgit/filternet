from mmengine.config import read_base
from filternet.datasets import LorenzDatasets
from filternet.datasets.transforms import MaxScaler
from filternet.models import DANSE
from filternet.params import LorenzParams
from filternet.trainer import CommonTrainer
with read_base():
    from .base import lorenz_params, DATA, METRIC, TRAINER, MONITOR, epoch, project, r2

lr = 5e-3
wd = 0
is_unsupervised = False

LOGGER = dict(project=project, name=f'Danse_GRU_is_unsupervised{is_unsupervised}', offline=False)

OPTIMIZER = dict(type='Adam', params=dict(lr=lr, weight_decay=wd))
SCHEDULER = dict(type='CommonScheduler', scheduler=dict(type='StepLR', params=dict(step_size=epoch // 6, gamma=0.9)))

LOSS = dict(
    loss_name='LogPDFGaussian',
    params=dict(reduction='none'),
    pred_loss_mask=None,
    target_loss_mask=None,
)
MODEL = dict(
    type=DANSE,
    params=lorenz_params,
    loss_cfg=LOSS,
    seq_len=100,
    covar_r=f'{r2}',
    # covar_r=1,
    rnn_params={
        'model_type': 'gru',
        'n_hidden': 30,
        'n_layers': 1,
        'n_hidden_dense': 32, },
    is_unsupervised=is_unsupervised,
)
