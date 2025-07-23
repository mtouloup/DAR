import numpy as np

from cutsimulator.cluster.cluster import Cluster
from cutsimulator.reward.reward import BaseReward
from cutsimulator.utils.utility import safe_ratio

class LB_reward(BaseReward):
    def __init__(self, cluster):
        self.cluster = cluster

    def compute(self, selected_node, valid_nodes):
        rewards_list = []

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

        std_cpu = np.std(cpu_usages)
        std_mem = np.std(mem_usages)
        cluster_load_score = max(0, 1 - (std_cpu + std_mem) / 2)

        node_balance_scores = []
        for cpu_usage, mem_usage in zip(cpu_usages, mem_usages):
            balance = max(0, 1 - abs(cpu_usage - mem_usage))
            node_balance_scores.append(balance)

        idle_penalty = safe_ratio(node_has_pods.count(0), len(all_nodes))

        for node in valid_nodes:
            reward = 0

            if node == selected_node:
                reward += 1

            reward += cluster_load_score

            try:
                node_index = all_nodes.index(node)
                reward += node_balance_scores[node_index]
            except ValueError:
                pass

            reward -= idle_penalty

            rewards_list.append(reward)

        return rewards_list

    def onClusterReset(self, cluster: Cluster):
        self.cluster = cluster
