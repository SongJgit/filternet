import argparse
import os
import os.path as osp
from typing import Dict

from lightning import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from mmengine.config import Config, DictAction

from filternet.datasets import CommonDataModule
from filternet.registry import MODELS
from filternet.utils import generate_save_dir, training_info


def main(args: argparse.ArgumentParser, cfg: Config) -> None:
    training_info()

    save_dir: Dict = generate_save_dir(root='./runs',
                                       project=cfg.LOGGER.project,
                                       name=cfg.LOGGER.name)
    cfg.LOGGER.name = save_dir['new_name']
    seed_everything(cfg.TRAINER.random_seed)

    data_module = CommonDataModule(cfg)
    data_module.setup()
    # model.

    model = MODELS.build(
        dict(type=cfg.TRAINER.type, cfg=cfg, save_dir=save_dir))
    # trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # model_monitor = ModelCheckpoint(
    #     dirpath=save_dir['weight_dir'],
    #     filename='epoch={epoch}-val_RMSE={val/RMSE:.5f}',
    #     mode='min',
    #     save_top_k=-1,
    #     auto_insert_metric_name=False,
    #     monitor='val/RMSE')
    model_monitor = ModelCheckpoint(
        dirpath=save_dir['weight_dir'],
        filename='epoch={epoch}-val_mAP={val/mAP:.5f}',
        mode='max',
        save_top_k=-1,
        auto_insert_metric_name=False,
        monitor='val/mAP')

    callbacks = [lr_monitor, model_monitor]
    wandb_logger = WandbLogger(project=cfg.LOGGER.project,
                               name=cfg.LOGGER.name,
                               offline=cfg.LOGGER.offline,
                               config=cfg)
    trainer = Trainer(
        accelerator=cfg.TRAINER.accelerator,
        max_epochs=cfg.TRAINER.epochs,
        logger=wandb_logger,
        log_every_n_steps=1,
        detect_anomaly=cfg.TRAINER.detect_anomaly,
        callbacks=callbacks,
        devices=cfg.TRAINER.device,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg.TRAINER.check_val_every_n_epoch
        if cfg.TRAINER.check_val_every_n_epoch is not None else 1
        # manual gradient clip
        # gradient_clip_val=cfg.TRAINER.gradient_clip_val,
        # gradient_clip_algorithm=cfg.TRAINER.gradient_clip_algorithm,
    )

    trainer.fit(model, datamodule=data_module)
    if not osp.exists(save_dir['config_dir']):
        os.makedirs(save_dir['config_dir'])
    cfg.dump(osp.join(save_dir['config_dir'], 'config.py'))

    # trainer.test(ckpt_path='best', datamodule=data_module)


def parse_args():
    parser = argparse.ArgumentParser(
        prog='filternet',
        description='Dataset, training and network parameters')
    parser.add_argument('--config',
                        '--cfg',
                        type=str,
                        metavar='config',
                        help='model and seq ')

    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_known_args()[0]
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    print(cfg.pretty_text)
    main(args, cfg)
