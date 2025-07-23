from typing import List

from cutsimulator.cluster.node import Node
from cutsimulator.cluster.cluster import Cluster
from cutsimulator.cluster.kwok_cluster import KWOKCluster
from cutsimulator.cluster.python_cluster import PythonCluster
from cutsimulator.utils.utility import generate_distribution_values

# Cluster synthesizer is responsible for creating a cluster with a set of nodes.
# The node characteristics (e.g., cpu, mem) are generated based on the provided
# config settings.
class ClusterSynthesizer:
    def __init__(self, config):
        required_keys = ['cluster_type', 'cluster_reset', 'cluster_nodes', 
                         'cluster_nodes_cpu_dist', 'cluster_nodes_mem_dist']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required cluster config key: {key}")
        self.config = config

    def create_nodes(self, start_index=0) -> List[Node]:
        num_nodes = self.config['cluster_nodes']
        cpu_dist = self.config['cluster_nodes_cpu_dist']
        mem_dist = self.config['cluster_nodes_mem_dist']

        cpus = generate_distribution_values(cpu_dist, num_nodes)
        memories = generate_distribution_values(mem_dist, num_nodes)

        nodes = []
        for i in range(num_nodes):
            nodes.append(Node(f"node-{start_index + i + 1}", cpus[i], memories[i]))

        return nodes

    def create_cluster(self) -> Cluster:
        # Create the appropriate cluster
        cluster_type = self.config['cluster_type']
        if cluster_type == 'KWOK':
            cluster = KWOKCluster()
        elif cluster_type == 'Python':
            cluster = PythonCluster()
        else:
            raise ValueError(f"Unsupported cluster type {cluster_type}")
        
        # Reset the cluster                             
        cluster_reset = self.config['cluster_reset']
        if cluster_reset:
            cluster.reset()
            start_index = 0
        else:
            start_index = cluster.get_num_nodes()
        
        # Create and deploy new nodes
        nodes = self.create_nodes(start_index)
        cluster.deploy_nodes(nodes)

        return cluster
