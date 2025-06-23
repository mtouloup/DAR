from scheduler.scheduler import Scheduler
from scheduler.daro_scheduler import DAROScheduler
from scheduler.default_scheduler import DefaultScheduler
from scheduler.round_robin_scheduler import RoundRobinScheduler

class SchedulerSelector:
    def __init__(self, config):
        if 'Scheduler' not in config:
            raise ValueError("SchedulerSelector requires the 'Scheduler' configuration")
        self.config = config

    def create_scheduler(self, cluster) -> Scheduler:
        scheduler_type = self.config["Scheduler"]["Type"]

        if scheduler_type == "DARO":
            scheduler = DAROScheduler(self.config, cluster)
        elif scheduler_type == "ROUNDROBIN":
            scheduler = RoundRobinScheduler(self.config, cluster)
        elif scheduler_type == "DEFAULT":
            scheduler = DefaultScheduler(self.config, cluster)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        return scheduler
    