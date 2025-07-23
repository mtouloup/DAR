import numpy as np

from cutsimulator.cluster.cluster import Cluster
from cutsimulator.reward.reward import BaseReward
from cutsimulator.utils.utility import safe_ratio


class Coop_LB_reward(BaseReward): # Cooprative Load Balancing Reward
    def __init__(self, cluster):
        self.cluster = cluster

    def compute(self, selected_node, valid_nodes):
        all_nodes = self.cluster.get_nodes()
        cpu_usages = []
        mem_usages = []
        node_has_pods = []

        for node in all_nodes:
            cpu_usage = 1 - safe_ratio(node.cpu_available, node.cpu_capacity)
            mem_usage = 1 - safe_ratio(node.mem_available, node.mem_capacity)
            cpu_usages.append(cpu_usage)
            mem_usages.append(mem_usage)
            participated = int(cpu_usage > 0.01 or mem_usage > 0.01)
            node_has_pods.append(participated)

        # Cluster-wide load uniformity (low stddev = better)
        std_cpu = np.std(cpu_usages)
        std_mem = np.std(mem_usages)
        cluster_load_score = max(0, 1 - (std_cpu + std_mem) / 2)

        # Node-local CPU/mem balance
        node_balance_scores = [max(0, 1 - abs(c - m)) for c, m in zip(cpu_usages, mem_usages)]
        avg_node_balance = np.mean(node_balance_scores)

        # Idle penalty
        idle_penalty = safe_ratio(node_has_pods.count(0), len(all_nodes))

        # Base reward formula (cooperative scalar)
        reward = cluster_load_score + avg_node_balance - idle_penalty

        # Optional: add pod success bonus (only if selected_node is valid)
        if selected_node in valid_nodes:
            reward += 0.5  # encourage successful scheduling

        # Return the same reward for all agents
        return [reward for _ in valid_nodes]

    def onClusterReset(self, cluster: Cluster):
        self.cluster = cluster
