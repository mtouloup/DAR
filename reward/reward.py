from abc import ABC, abstractmethod
from cluster.cluster import Cluster
from cluster.node import Node
from typing import List

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
