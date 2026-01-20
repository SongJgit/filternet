import logging
from lightning.pytorch.callbacks.callback import Callback
from typing import Any, Union
import lightning.pytorch as pl
import torch
from typing_extensions import override
from filternet.utils import model_summary, model_grad_graph

log = logging.getLogger(__name__)


class ModelSummary(Callback):

    def __init__(self, max_depth: int = 5, create_graph: bool = False, save_dir: str | None = None):
        self.max_depth = max_depth
        self.create_graph = create_graph
        self.save_dir = save_dir

        if self.create_graph:
            assert self.save_dir is not None, 'save_dir must be specified if create_graph is True'

    @override
    def on_fit_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if trainer.is_global_zero:
            #
            model_summary(pl_module.model, depth=self.max_depth)

            if self.create_graph:
                model_grad_graph(pl_module.model, save_dir=self.save_dir)
