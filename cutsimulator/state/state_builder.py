import numpy as np

import cutsimulator.state.feature_builder as fb
from cutsimulator.cluster.cluster import Cluster
from cutsimulator.utils.utility import safe_ratio
from cutsimulator.workload.pod import Pod

# Returns the number of dimensions in the state.
# It must match the number of elements returned by build_node_state()
def state_dimensions(cluster: Cluster, num_max_agents: int = None) -> int:
    return fb.cluster_features_dimensions() + fb.node_features_dimensions() * (cluster.get_num_nodes() if num_max_agents is None else num_max_agents)

# Builds the current state based on the cluster, nodes, and pod
def build_cluster_state(cluster: Cluster, pod: Pod) -> np.ndarray:
    
    state = fb.build_cluster_features(cluster)

    for node in cluster.get_nodes():
        state = state + fb.build_node_features(cluster, node, pod)

    return np.array(state, dtype=np.float32)
