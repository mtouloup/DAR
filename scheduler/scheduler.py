from abc import ABC, abstractmethod
from cluster.cluster import Cluster
from cluster.node import Node
from workload.pod import Pod

class Scheduler(ABC):
    
    @abstractmethod
    def schedule(self, pod: Pod) -> Node:
        pass

    @abstractmethod
    def onPodDeployed(self, pod: Pod):
        pass

    @abstractmethod
    def onPodTerminated(self, pod: Pod):
        pass

    @abstractmethod
    def onSimulationEnded(self):
        pass

    @abstractmethod
    def onClusterReset(self, cluster: Cluster):
        pass
