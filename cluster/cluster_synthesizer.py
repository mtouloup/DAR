from typing import List
from cluster.node import Node
from cluster.cluster import Cluster
from cluster.kwok_cluster import KWOKCluster
from cluster.python_cluster import PythonCluster
from utils.utility import generate_distribution_values

# Cluster synthesizer is responsible for creating a cluster with a set of nodes.
# The node characteristics (e.g., cpu, mem) are generated based on the provided
# config settings.
class ClusterSynthesizer:
    def __init__(self, config):
        if 'Cluster' not in config:
            raise ValueError("ClusterSynthesizer requires the 'Cluster' configuration")
        self.config = config

    def create_nodes(self, start_index=0) -> List[Node]:
        num_nodes = self.config['Cluster']['Nodes']['Number']
        cpu_dist = self.config['Cluster']['Nodes']['CPU Dist']
        mem_dist = self.config['Cluster']['Nodes']['Mem Dist']

        cpus = generate_distribution_values(cpu_dist, num_nodes)
        memories = generate_distribution_values(mem_dist, num_nodes)

        nodes = []
        for i in range(num_nodes):
            nodes.append(Node(f"node-{start_index + i + 1}", cpus[i], memories[i]))

        return nodes

    def create_cluster(self) -> Cluster:
        # Create the appropriate cluster
        cluster_type = self.config['Cluster']['Type']
        if cluster_type == 'KWOK':
            cluster = KWOKCluster()
        elif cluster_type == 'Python':
            cluster = PythonCluster()
        else:
            raise ValueError(f"Unsupported cluster type {cluster_type}")
        
        # Reset the cluster                             
        cluster_reset = self.config['Cluster']['Reset']
        if cluster_reset:
            cluster.reset()
            start_index = 0
        else:
            start_index = cluster.get_num_nodes()
        
        # Create and deploy new nodes
        nodes = self.create_nodes(start_index)
        cluster.deploy_nodes(nodes)

        return cluster
