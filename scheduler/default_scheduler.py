from cluster.cluster import Cluster, Node
from scheduler.scheduler import Scheduler
from workload.pod import Pod

# A scheduler that lets the default KWOK scheduler perform the scheduling decisions
class DefaultScheduler(Scheduler):
    def __init__(self):
        pass  # No setup needed for default scheduling

    def schedule(self, pod: Pod) -> Node:
        pass

    def onPodDeployed(self, pod: Pod):
        pass

    def onPodTerminated(self, pod: Pod):
        pass

    def onSimulationEnded(self):
        pass

    def onClusterReset(self, cluster: Cluster):
        pass
