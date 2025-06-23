import random
import torch
import numpy as np
import random as rnd
from cluster.cluster import Cluster
from utils.utility import safe_ratio, log_rewards
from scheduler.qmix_agent import QMIX
from workload.pod import Pod

class Broker:
    def __init__(self, cluster, reward_fn, num_agents, 
                 input_dim=8, output_dim=10, hidden_dim=64, lr=0.001, gamma=0.99,
                 update_target_every=200, double_q=True, epsilon=0.1, mixing_embed_dim=32, 
                 hypernet_layers=2, hypernet_embed=64, buffer_size=1000, batch_size=32):
        self.cluster = cluster  # Broker uses Cluster object
        self.num_agents = num_agents
        self.output_dim = output_dim + 1 # Always 11 (10 bids + no-op)
        self.input_dim = input_dim  # Always 8 features (cluster + node + pod metrics)

        self.cache = {} # Cache info until a pod is actually scheduled

        self.qmix = QMIX(num_agents=self.num_agents, input_dim=self.input_dim, output_dim=self.output_dim, 
                         hidden_dim=hidden_dim, lr=lr, gamma=gamma, update_target_every=update_target_every, 
                         double_q=double_q, mixing_embed_dim=mixing_embed_dim, 
                         hypernet_layers=hypernet_layers, hypernet_embed=hypernet_embed)
        self.epsilon=epsilon
        self.replay_buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.reward_fn = reward_fn

    def build_node_state(self, node, pod):
        cluster_state = self.cluster.get_cluster_state()

        # Extract cluster-wide info
        cluster_cpu_capacity = cluster_state['total_cpu_capacity']
        cluster_mem_capacity = cluster_state['total_mem_capacity']
        cluster_cpu_available = cluster_state['total_cpu_available']
        cluster_mem_available = cluster_state['total_mem_available']

        # Extract node-specific info
        node_cpu_capacity = node.cpu_capacity
        node_mem_capacity = node.mem_capacity
        node_cpu_available = node.cpu_available
        node_mem_available = node.mem_available

        # Pod requirements
        pod_cpu = pod.cpu
        pod_mem = pod.memory

        # Build normalized state
        cluster_cpu_ratio = safe_ratio(cluster_cpu_available, cluster_cpu_capacity)
        cluster_mem_ratio = safe_ratio(cluster_mem_available, cluster_mem_capacity)
        
        node_cpu_ratio = safe_ratio(node_cpu_available, node_cpu_capacity)
        node_mem_ratio = safe_ratio(node_mem_available, node_mem_capacity)
        
        cpu_cluster_node = safe_ratio(node_cpu_capacity, cluster_cpu_capacity)
        mem_cluster_node = safe_ratio(node_mem_capacity, cluster_mem_capacity)
        
        request_cpu_ratio = safe_ratio(pod_cpu, node_cpu_available)
        request_mem_ratio = safe_ratio(pod_mem, node_mem_available)
        
        state = [
            cluster_cpu_ratio, cluster_mem_ratio,
            node_cpu_ratio, node_mem_ratio,
            cpu_cluster_node, mem_cluster_node,
            request_cpu_ratio, request_mem_ratio
        ]

        return np.array(state, dtype=np.float32)

        
    def schedule_pod(self, pod):
        nodes = self.cluster.get_nodes()

        # Remove nodes that don't have enough resources
        valid_nodes = [node.has_available_resources(pod.cpu, pod.memory) for node in nodes]

        if not np.any(valid_nodes):
            print(f"[Broker] No valid nodes found for Pod {pod.name}")
            return None  # No node can schedule this pod

        # Build states
        states = np.array([self.build_node_state(node, pod) for node in nodes])

        # Select actions (bids)
        actions = self.qmix.select_actions(states, valid_nodes, epsilon=self.epsilon)

        # Pick the node with highest bid
        max_bid = max(actions)
        best_nodes = [node for node, bid in zip(nodes, actions) if bid == max_bid]
        selected_node = random.choice(best_nodes)

        # Cache the info until the pod is actually scheduled
        self.cache[pod.name] = (states, actions)

        print(f"[Broker] Pod {pod.name} scheduled on {selected_node.name} with bid {max_bid}")

        return selected_node

    def onPodDeployed(self, pod: Pod):
        if pod.name not in self.cache:
            print(f"[Broker] Cache not found for Pod {pod.name}")
            return
            
        # Build next state
        nodes = self.cluster.get_nodes()
        next_states = np.array([self.build_node_state(node, pod) for node in nodes])

        # Compute reward
        rewards = self.reward_fn.compute(pod.node, nodes)
        log_rewards(pod.name, pod.node, nodes, rewards)
        
        # Save the experience for training
        self.replay_buffer.append((self.cache[pod.name][0], self.cache[pod.name][1], np.mean(rewards), next_states))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
        del self.cache[pod.name]

        # Train
        if len(self.replay_buffer) >= self.batch_size:
            batch = rnd.sample(self.replay_buffer, self.batch_size)
            self.qmix.train(batch)
            print(f"[Broker] QMIX training updated.")

    def save_model(self, path="qmix_latest.pth"):
        torch.save(self.qmix, path)
        print(f"[Broker] Model saved to {path}")
    
    def onClusterReset(self, cluster: Cluster):
        self.cluster = cluster
        self.reward_fn.onClusterReset(cluster)
        self.cache.clear()
