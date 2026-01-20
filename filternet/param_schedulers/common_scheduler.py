from typing import Any, Dict

from torch.optim import lr_scheduler  # noqa: F401

from filternet.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class CommonScheduler:

    def __init__(self, optimizer, scheduler: Dict[str, Any]):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def get_scheduler(self):
        return eval(f'lr_scheduler.{self.scheduler.type}')(optimizer=self.optimizer, **self.scheduler.params)
