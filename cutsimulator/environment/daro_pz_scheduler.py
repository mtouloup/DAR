import numpy as np
import random as rnd

import cutsimulator.state.obs_builder as ob
from cutsimulator.cluster.cluster import Cluster, Node
from cutsimulator.environment.coordinator import Coordinator
from cutsimulator.reward.reward_selector import RewardSelector
from cutsimulator.scheduler.scheduler import Scheduler
from cutsimulator.utils.utility import log_rewards
from cutsimulator.workload.pod import Pod

# A scheduler to be used with the Daro PettingZoo environment
class DaroPettingZooScheduler(Scheduler):
    def __init__(self, config: dict, coordinator: Coordinator):
        self.reward_fn = RewardSelector(config, None).create_reward()
        self.coordinator = coordinator

    def schedule(self, pod: Pod) -> Node:
        nodes = self.cluster.get_nodes()

        # Mark nodes that don't have enough resources
        self.valid_nodes = [node.has_available_resources(pod.cpu, pod.memory) for node in nodes]
        if not np.any(self.valid_nodes):
            print(f"[Scheduler] No valid nodes found for Pod {pod.name}")
            return None  # No node can schedule this pod

        # Build states and switch to the environment to select actions
        self.obs = {f"agent_{i}": ob.build_node_obs(self.cluster, node, pod) for i, node in enumerate(nodes)}
        self.coordinator.switch_turn()
        self.coordinator.wait_for_turn(is_main=False)

        # Pick the node with highest bid
        max_bid = max(self.actions.values())
        best_nodes = [node for node, bid in zip(nodes, self.actions.values()) if bid == max_bid]
        selected_node = rnd.choice(best_nodes)

        print(f"[Scheduler] Pod {pod.name} scheduled on {selected_node.name} with bid {max_bid}")

        return selected_node

    def onPodDeployed(self, pod: Pod):
        # Compute reward
        nodes = self.cluster.get_nodes()
        reward_list = self.reward_fn.compute(pod.node, nodes)
        self.rewards = {f"agent_{i}": reward for i, reward in enumerate(reward_list)}
        log_rewards(pod.name, pod.node, nodes, reward_list)

    def onPodTerminated(self, pod: Pod):
        pass

    def onSimulationEnded(self):
        # Mark simulation end and switch to the environment
        self.sim_running = False
        self.coordinator.switch_turn()

    def onClusterReset(self, cluster: Cluster):
        # A new simulation is about to start
        self.sim_running = True
        self.cluster = cluster
        self.reward_fn.onClusterReset(cluster)
        self.rewards = {f"agent_{i}": 0 for i in range(len(cluster.get_nodes()))}

    def isSimRunning(self):
        return self.sim_running

    def setActions(self, actions):
        self.actions = actions
        # Ensure the actions are appropriate
        for i, agent in enumerate(self.actions):
            if not self.valid_nodes[i]:
                self.actions[agent] = 0
            elif self.actions[agent] == 0:
                self.actions[agent] = 1

    def getObservations(self):
        return self.obs
    
    def getRewards(self):
        return self.rewards
    
    def getValidNodes(self):
        return self.valid_nodes

    