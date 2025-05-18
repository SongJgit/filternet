from filternet.datasets import LorenzDatasets
from filternet.trainer import CommonTrainer
from filternet.params import LorenzParams

q2 = '0.0001'
r2 = '1000.0'
date = '25050716'

if r2 == '1.0':
    rmse = '2.31'
elif r2 == '10.0':
    rmse = '3.78'
elif r2 == '100.0':
    rmse = '10.26'
elif r2 == '1000.0':
    rmse = '31.56'

data_path = f'./data/lorenz_data/{date}/LorenzSSM_3x3_T100_N1000_q2_{q2}_r2_{r2}_TestObsRMSE{rmse}.pkl'
project = f'LorenzSSM_3x3_T100_N1000_q2_{q2}_r2_{r2}_TestObsRMSE{rmse}'
batch_size = 64
epoch = 100
seed = 3407

train_seq_len = None
use_ntrain = None
test_seq_len = None
use_ntest = None
val_seq_len = None
use_nval = None
drop_last = False

slide_win_size = None
detach_step = None

grad_clip_val = 10

device = [0]  # use cuda:0

transforms = None
TRAIN_DATASET = dict(
    type=LorenzDatasets,
    data_path=data_path,
    mode='train',
    seq_len=train_seq_len,
    split_data=False,
    use_ndata=use_ntrain,
    transforms=transforms,
    drop_last=drop_last,
)

VAL_DATASET = dict(
    type=LorenzDatasets,
    data_path=data_path,
    mode='val',
    seq_len=None,
    split_data=False,
    use_ndata=use_nval,
    transforms=transforms,
    drop_last=drop_last,
)

TEST_DATASET = dict(
    type=LorenzDatasets,
    data_path=data_path,
    mode='test',
    seq_len=None,
    split_data=False,
    use_ndata=use_ntest,
    transforms=transforms,
    drop_last=drop_last,
)

DATA = dict(train_dataset=TRAIN_DATASET,
            val_dataset=VAL_DATASET,
            test_dataset=TEST_DATASET,
            train_batch_size=batch_size,
            test_batch_size=batch_size,
            val_batch_size=batch_size,
            shuffle=False,
            num_workers=1)

lorenz_params = dict(type=LorenzParams, dim_state=3, j=5, dt=0.02)

METRIC = dict(pred_metric_mask=None,
              target_metric_mask=None,
              num_metric_dims=None,
              metric_dim_names=None,
              inverse_transforms=False)

TRAINER = dict(
    type=CommonTrainer,
    accelerator='gpu',
    epochs=epoch,
    detach_step=detach_step,
    batch_size=batch_size,
    gradient_clip_val=grad_clip_val,  # int | float | None
    gradient_clip_algorithm='norm',  # str | None
    slide_win_size=slide_win_size,
    device=device,
    detect_anomaly=False,
    check_val_every_n_epoch=5,
    random_seed=seed,
    test=True,
)

MONITOR = dict(MODEL_MONITOR=dict(filename='epoch={epoch}-val_RMSE={val/RMSE:.5f}',
                                  mode='min',
                                  save_top_k=-1,
                                  auto_insert_metric_name=False,
                                  monitor='val/RMSE'))
