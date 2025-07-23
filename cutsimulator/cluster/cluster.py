from abc import ABC, abstractmethod
from typing import List

from cutsimulator.cluster.node import Node
from cutsimulator.workload.pod import Pod

class Cluster(ABC):
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def deploy_nodes(self, nodes: List[Node]):
        pass

    @abstractmethod
    def get_nodes(self) -> List[Node]:
        pass

    @abstractmethod
    def get_num_nodes(self) -> int:
        pass

    @abstractmethod
    def deploy_pod(self, pod: Pod, node: Node) -> bool:
        pass

    @abstractmethod
    def terminate_pod(self, pod: Pod) -> bool:
        pass

    @abstractmethod
    def get_pod_node(self, pod_name: str) -> Node:
        pass

    @abstractmethod
    def get_node(self, node_name: str) -> Node:
        pass

    def get_cluster_state(self):
        nodes = self.get_nodes()
        total_cpu_capacity = sum(n.cpu_capacity for n in nodes)
        total_mem_capacity = sum(n.mem_capacity for n in nodes)
        total_cpu_available = sum(n.cpu_available for n in nodes)
        total_mem_available = sum(n.mem_available for n in nodes)

        return {"total_cpu_capacity" : total_cpu_capacity, 
                "total_mem_capacity" : total_mem_capacity,
                "total_cpu_available" : total_cpu_available,
                "total_mem_available" : total_mem_available}

