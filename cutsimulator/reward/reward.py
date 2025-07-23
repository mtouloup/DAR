from abc import ABC, abstractmethod
from typing import List

from cutsimulator.cluster.cluster import Cluster
from cutsimulator.cluster.node import Node

class BaseReward(ABC):
    @abstractmethod
    def compute(self, selected_node: Node, valid_nodes: List[Node]) -> List[float]:
        """
        Computes reward for all valid nodes and return list of rewards 
        """
        pass

    @abstractmethod
    def onClusterReset(self, cluster: Cluster):
        pass
