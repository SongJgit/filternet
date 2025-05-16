from filternet.datasets import NCLTFusionDataset, nclt_collate
from filternet.trainer import NCLTFusionTrainer
from filternet.params import NCLTFusionWheelGPSParams

train_path = './data/nclt_fusion/train.pt'
test_path = './data/nclt_fusion/test.pt'
val_path = './data/nclt_fusion/val.pt'
val_path = [val_path]

train_seq_len = 50
batch_size = 256
slide_win_size = None
detach_step = None
epoch = 50
grad_clip_val = 1
device = [0]
seed = 3407

project = 'nclt_fusion'

params = dict(type=NCLTFusionWheelGPSParams, dt=1, dim_state=6, obs_ind=[0, 1], noise_q2=1, noise_r2=1)

TRAIN_DATASET = dict(
    type=NCLTFusionDataset,
    data_path=train_path,
    seq_len=train_seq_len,
)

VAL_DATASET = dict(
    type=NCLTFusionDataset,
    data_path=val_path,
    seq_len=None,
)

TEST_DATASET = dict(
    type=NCLTFusionDataset,
    data_path=test_path,
    seq_len=None,
)

DATA = dict(train_dataset=TRAIN_DATASET,
            val_dataset=VAL_DATASET,
            test_dataset=TEST_DATASET,
            train_batch_size=batch_size,
            test_batch_size=batch_size,
            val_batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=nclt_collate)

METRIC = dict(
    pred_metric_mask=[True, True, False, False, False, False],
    target_metric_mask=[True, True],  # must match state
    # number of state to compute metric ,metric_dims must be equal to number of mask'True
    num_metric_dims=2,
    metric_dim_names=['X', 'Y'],
    inverse_transforms=False)

TRAINER = dict(
    type=NCLTFusionTrainer,
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
