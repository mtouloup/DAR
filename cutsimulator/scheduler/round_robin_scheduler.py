from cutsimulator.cluster.cluster import Cluster
from cutsimulator.cluster.node import Node
from cutsimulator.scheduler.scheduler import Scheduler
from cutsimulator.workload.pod import Pod

# Schedules pod on the available nodes in a round-robin fashion
class RoundRobinScheduler(Scheduler):
    def __init__(self, config, cluster: Cluster):
        self.cluster = cluster
        self.last_node_idx = -1

    def schedule(self, pod: Pod) -> Node:
        nodes = self.cluster.get_nodes()
        num_nodes = len(nodes)

        for i in range(num_nodes):
            # Find the next node with available resources
            self.last_node_idx = (self.last_node_idx + 1) % num_nodes
            if nodes[self.last_node_idx].has_available_resources(pod.cpu, pod.memory):
                return nodes[self.last_node_idx]
        
        # If we reach this point, then no node has available resources
        return None

    def onPodDeployed(self, pod: Pod):
        pass

    def onPodTerminated(self, pod: Pod):
        pass 

    def onSimulationEnded(self):
        pass

    def onClusterReset(self, cluster: Cluster):
        self.cluster = cluster
        self.last_node_idx = -1
