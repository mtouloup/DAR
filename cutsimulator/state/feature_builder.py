import numpy as np

from cutsimulator.cluster.cluster import Cluster, Node
from cutsimulator.workload.pod import Pod
from cutsimulator.utils.utility import safe_ratio

# Returns the number of cluster features returned by build_cluster_features()
def cluster_features_dimensions() -> int:
    return 2

# Builds the cluster-related features
def build_cluster_features(cluster: Cluster) -> list:
    cluster_state = cluster.get_cluster_state()

    # Extract cluster-wide info
    cluster_cpu_capacity = cluster_state["total_cpu_capacity"]
    cluster_mem_capacity = cluster_state["total_mem_capacity"]
    cluster_cpu_available = cluster_state["total_cpu_available"]
    cluster_mem_available = cluster_state["total_mem_available"]

    # Build normalized features
    cluster_cpu_ratio = safe_ratio(cluster_cpu_available, cluster_cpu_capacity)
    cluster_mem_ratio = safe_ratio(cluster_mem_available, cluster_mem_capacity)
    
    features = [
        cluster_cpu_ratio, cluster_mem_ratio,
    ]

    return features

# Returns the number of node features returned by build_node_features()
def node_features_dimensions() -> int:
    return 6

# Builds the node-related features
def build_node_features(cluster: Cluster, node: Node, pod: Pod) -> list:
    cluster_state = cluster.get_cluster_state()

    # Extract cluster-wide info
    cluster_cpu_capacity = cluster_state["total_cpu_capacity"]
    cluster_mem_capacity = cluster_state["total_mem_capacity"]

    # Extract node-specific info
    node_cpu_capacity = node.cpu_capacity
    node_mem_capacity = node.mem_capacity
    node_cpu_available = node.cpu_available
    node_mem_available = node.mem_available

    # Pod requirements
    pod_cpu = pod.cpu
    pod_mem = pod.memory

    # Build normalized features
    node_cpu_ratio = safe_ratio(node_cpu_available, node_cpu_capacity)
    node_mem_ratio = safe_ratio(node_mem_available, node_mem_capacity)
    
    cpu_cluster_node = safe_ratio(node_cpu_capacity, cluster_cpu_capacity)
    mem_cluster_node = safe_ratio(node_mem_capacity, cluster_mem_capacity)
    
    request_cpu_ratio = safe_ratio(pod_cpu, node_cpu_available)
    request_mem_ratio = safe_ratio(pod_mem, node_mem_available)
    
    features = [
        node_cpu_ratio, node_mem_ratio,
        cpu_cluster_node, mem_cluster_node,
        request_cpu_ratio, request_mem_ratio
    ]

    return features
