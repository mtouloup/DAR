from typing import List

from cutsimulator.cluster.node import Node
from cutsimulator.cluster.cluster import Cluster
from cutsimulator.workload.pod import Pod
import logging
logger = logging.getLogger(__name__)

# Simulates a virtual cluster in Python
class PythonCluster(Cluster):
    def __init__(self):
        self.nodes = []
        self.pods = {}

    def reset(self):        
        logger.info("Resetting PythonCluster...")
        self.nodes.clear()
        self.pods.clear()
    
    def deploy_nodes(self, nodes: List[Node]):
        self.nodes.extend(nodes)
        for node in nodes:
            logger.info(f"Added cluster node {node}")

    def get_nodes(self) -> List[Node]:
        return self.nodes

    def get_num_nodes(self) -> int:
        return len(self.nodes)

    def deploy_pod(self, pod: Pod, node: Node) -> bool:
        if node is None:
            logger.warning(f"Cannot deploy pod {pod.name} - no node provided")
            return False
        
        if not node.has_available_resources(pod.cpu, pod.memory):
            logger.warning(f"Cannot deploy pod {pod.name} - node has not enough resources")
            return False

        # Deploy the pod on the provided node
        node.allocate_resources(pod.cpu, pod.memory)
        pod.node = node
        self.pods[pod.name] = node

        return True

    def terminate_pod(self, pod: Pod) -> bool:
        if pod.name not in self.pods:
            logger.warning(f"Unable to terminate pod {pod.name} - not found")
            return False

        node = self.pods[pod.name]
        node.release_resources(pod.cpu, pod.memory)
        del self.pods[pod.name]
        return True

    def get_pod_node(self, pod_name: str) -> Node:
        return self.pods[pod_name]
    
    def get_node(self, node_name: str) -> Node:
        return next((node for node in self.nodes if node.name == node_name), None)

