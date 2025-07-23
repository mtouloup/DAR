import numpy as np

import cutsimulator.state.feature_builder as fb
from cutsimulator.cluster.cluster import Cluster, Node
from cutsimulator.workload.pod import Pod
from cutsimulator.utils.utility import safe_ratio

# Returns the number of dimensions in the observations.
# It must match the number of elements returned by build_node_obs()
def obs_dimensions() -> int:
    return fb.cluster_features_dimensions() + fb.node_features_dimensions()

# Builds the current observation based on the cluster, node, and pod
def build_node_obs(cluster: Cluster, node: Node, pod: Pod) -> np.ndarray:

    obs = fb.build_cluster_features(cluster)
    obs += fb.build_node_features(cluster, node, pod)

    return np.array(obs, dtype=np.float32)
