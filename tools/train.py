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
from filternet.utils import global_logger as netlogger, ModelSummary

from swanlab.integration.pytorch_lightning import SwanLabLogger


def main(args: argparse.ArgumentParser, cfg: Config) -> None:
    training_info()

    save_dir: Dict = generate_save_dir(root='./runs', project=cfg.LOGGER.project, name=cfg.LOGGER.name)
    cfg.LOGGER.name = save_dir['exp_name']
    netlogger.info(f'Experiment name : {cfg.LOGGER.name}')
    seed_everything(cfg.TRAINER.random_seed)

    data_module = CommonDataModule(cfg)
    # data_module.setup()

    # model.
    model = MODELS.build(dict(type=cfg.TRAINER.type, cfg=cfg, save_dir=save_dir))
    # trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_monitor = ModelCheckpoint(dirpath=save_dir['weight_dir'], **cfg.MONITOR.MODEL_MONITOR)
    model_summary = ModelSummary(max_depth=3, create_graph=True, save_dir=save_dir['exp_dir'])
    callbacks = [lr_monitor, model_monitor, model_summary]

    if hasattr(cfg.LOGGER, 'type') and cfg.LOGGER.type == 'SwanLab':
        if cfg.LOGGER.offline:
            mode = 'local'
        else:
            mode = 'cloud'
        logger = SwanLabLogger(project=cfg.LOGGER.project,
                               experiment_name=cfg.LOGGER.name,
                               config=cfg,
                               resume=cfg.LOGGER.resume if hasattr(cfg.LOGGER, 'resume') else False,
                               description=cfg.LOGGER.description if hasattr(cfg.LOGGER, 'description') else '',
                               mode=mode)

    elif (hasattr(cfg.LOGGER, 'type') and cfg.LOGGER.type == 'Wandb') or not hasattr(cfg.LOGGER, 'type'):
        # wandb is the default logger
        logger = WandbLogger(project=cfg.LOGGER.project, name=cfg.LOGGER.name, offline=cfg.LOGGER.offline, config=cfg)

    trainer = Trainer(accelerator=cfg.TRAINER.accelerator,
                      max_epochs=cfg.TRAINER.epochs,
                      logger=logger,
                      log_every_n_steps=1,
                      detect_anomaly=cfg.TRAINER.detect_anomaly,
                      callbacks=callbacks,
                      devices=cfg.TRAINER.device,
                      num_sanity_val_steps=0,
                      check_val_every_n_epoch=cfg.TRAINER.check_val_every_n_epoch
                      if cfg.TRAINER.check_val_every_n_epoch is not None else 1)

    trainer.fit(model, datamodule=data_module)
    if not osp.exists(save_dir['config_dir']):
        os.makedirs(save_dir['config_dir'])
    cfg.dump(osp.join(save_dir['config_dir'], 'config.py'))
    netlogger.info(f'📁 Training finished, best model saved at: {save_dir}')

    if hasattr(cfg.TRAINER, 'test') and cfg.TRAINER.test:
        trainer.test(ckpt_path='best', datamodule=data_module)


def parse_args():
    parser = argparse.ArgumentParser(prog='filternet', description='Dataset, training and network parameters')
    parser.add_argument('--config', '--cfg', type=str, metavar='config', help='model and seq ')

    parser.add_argument('--cfg_options',
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
