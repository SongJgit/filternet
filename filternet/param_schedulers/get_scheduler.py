from typing import Any, Dict

from torch.optim import lr_scheduler  # noqa: F401

from filternet.registry import PARAM_SCHEDULERS


def get_scheduler(cfg: Dict[str, Any]):
    scheduler = PARAM_SCHEDULERS.build(cfg.SCHEDULER)
    if not isinstance(scheduler, lr_scheduler.LRScheduler):
        # CommonScheduler
        scheduler = scheduler.get_scheduler()
    return scheduler
