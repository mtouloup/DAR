from cutsimulator.scheduler.scheduler import Scheduler
from cutsimulator.scheduler.daro_train_scheduler import DaroTrainScheduler
from cutsimulator.scheduler.default_scheduler import DefaultScheduler
from cutsimulator.scheduler.round_robin_scheduler import RoundRobinScheduler

class SchedulerSelector:
    def __init__(self, config):
        if 'scheduler_type' not in config:
            raise ValueError("SchedulerSelector requires 'scheduler_type' in config")
        self.config = config

    def create_scheduler(self, cluster) -> Scheduler:
        scheduler_type = self.config["scheduler_type"]

        if scheduler_type == "DAROTRAIN":
            scheduler = DaroTrainScheduler(self.config, cluster)
        elif scheduler_type == "ROUNDROBIN":
            scheduler = RoundRobinScheduler(self.config, cluster)
        elif scheduler_type == "DEFAULT":
            scheduler = DefaultScheduler()
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        return scheduler
    