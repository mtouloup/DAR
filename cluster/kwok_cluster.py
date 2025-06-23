import subprocess
import utils.utility as util
from kubernetes import client, config
from datetime import datetime
from typing import List
from cluster.node import Node
from cluster.cluster import Cluster
from workload.pod import Pod

# Represents a KWOK cluster
class KWOKCluster(Cluster):

    def __init__(self):
        try: 
            config.load_kube_config()  # Load kube config for external access
        except Exception as e:
            print(f"Failed to load config ({e}). Will try creating a kwok cluster first")
            subprocess.run(["kwokctl", "create", "cluster"], check=True)
            config.load_kube_config()

        self.api = client.CoreV1Api()

    def reset(self):
        """
        Resets the cluster using kwokctl.
        """
        print("Resetting cluster...")
        try:
            subprocess.run(["kwokctl", "delete", "cluster"], check=False)
            subprocess.run(["kwokctl", "create", "cluster"], check=True)
            print("Cluster successfully reset.")
        except Exception as e:
            print(f"Failed to reset cluster. Moving on. Error: {e}")

    def deploy_nodes(self, nodes: List[Node]):
        """
        Deploys nodes to the Kubernetes cluster using API.
        """
        for node in nodes:
            try:
                self.api.create_node(self._node_to_k8s_object(node))
                print(f"Successfully created node: {node.name}")
            except Exception as e:
                print(f"Failed to create node {node.name}: {e}")

    def get_nodes(self) -> List[Node]:
        """
        Retrieves a list of existing nodes in the cluster using Kubernetes API.
        """
        try:
            # Get all node info
            kwok_nodes = self.api.list_node()

            # Get all pods scheduled on the cluster
            kwok_pods = self.api.list_namespaced_pod(namespace="default")
        except Exception as e:
            print(f"Failed to get node info for the cluster: {e}")
            return []

        # Collect basic node info
        nodes = {}
        for node in kwok_nodes.items:
            if node.metadata.name.startswith("node-"):
                nodes[node.metadata.name] = Node(node.metadata.name, 
                        util.convert_cpu(node.status.allocatable.get("cpu", "0")), 
                        util.convert_memory(node.status.allocatable.get("memory", "0Mi")))
        
        # Compute the allocated resources
        for pod in kwok_pods.items:
            if pod.spec.node_name in nodes:
                for container in pod.spec.containers:
                    resources = container.resources.requests
                    if resources:
                        nodes[pod.spec.node_name].allocate_resources(
                            util.convert_cpu(resources.get("cpu", "0")),
                            util.convert_memory(resources.get("memory", "0")))

        return list(nodes.values())

    def get_num_nodes(self) -> int:
        """
        Retrieves the number of existing nodes in the cluster using Kubernetes API.
        """
        api = client.CoreV1Api()
        try:
            existing_nodes = api.list_node()
            return len(existing_nodes.items)
        except Exception as e:
            print(f"Failed to retrieve existing nodes. Assuming no nodes exist. Error: {e}")
            return 0

    def deploy_pod(self, pod: Pod, node: Node) -> bool:
        try:
            # Deploy on kwok
            pod_manifest = self._pod_to_k8s_object(pod, node)
            self.api.create_namespaced_pod(namespace="default", body=pod_manifest)

            # Make sure it was deployed
            pod.node = self.get_pod_node(pod.name)
            if pod.node is None:
                self.api.delete_namespaced_pod(name=pod.name, namespace="default")
                print(f"Unable to deploy pod {pod.name} on {node}")
                return False

            return True
        except Exception as e:
            print(f"Failed to deploy pod {pod.name}: {e}")
            return False

    def terminate_pod(self, pod: Pod) -> bool:
        try:
            self.api.delete_namespaced_pod(name=pod.name, namespace="default")
        except Exception as e:
            print(f"Failed to terminate pod {pod.name}: {e}")

    def get_pod_node(self, pod_name: str) -> Node:
        """
        Returns the node that this pod is running on.
        """
        try:
            kwok_pod = self.api.read_namespaced_pod(namespace="default", name=pod_name)
            if kwok_pod.spec.node_name:
                return self.get_node(kwok_pod.spec.node_name)
            else:
                return None
        except Exception as e:
            print(f"Failed to get node for pod {pod_name}: {e}")
            return None
        
    def get_node(self, node_name: str) -> Node:
        """
        Returns information for the given node name.
        """
        try:
            # Get basic node info
            kwok_node = self.api.read_node(name=node_name)

            # Get all pods scheduled on this node
            kwok_pods = self.api.list_namespaced_pod(namespace="default", field_selector=f"spec.nodeName={node_name}")
        except Exception as e:
            print(f"Failed to get node info for {node_name}: {e}")
            return None

        node = Node(node_name, 
                    util.convert_cpu(kwok_node.status.allocatable.get("cpu", "0")),
                    util.convert_memory(kwok_node.status.allocatable.get("memory", "0")))

        for pod in kwok_pods.items:
            for container in pod.spec.containers:
                resources = container.resources.requests
                if resources:
                    node.allocate_resources(util.convert_cpu(resources.get("cpu", "0")),
                                            util.convert_memory(resources.get("memory", "0")))
        
        return node


    def _node_to_k8s_object(self, node: Node):
        """
        Converts a Node instance to a Kubernetes API Node object.
        """
        k8s_node = client.V1Node(
            api_version="v1",
            kind="Node",
            metadata=client.V1ObjectMeta(
                name=node.name,
                labels={
                    "beta.kubernetes.io/arch": "amd64",
                    "beta.kubernetes.io/os": "linux",
                    "kubernetes.io/arch": "amd64",
                    "kubernetes.io/hostname": node.name,
                    "kubernetes.io/os": "linux",
                    "kubernetes.io/role": "agent",
                    "node-role.kubernetes.io/agent": "",
                    "type": "kwok",
                },
                annotations={"node.alpha.kubernetes.io/ttl": "0", "kwok.x-k8s.io/node": "fake"},
            ),
            spec=client.V1NodeSpec(
                taints=[
                    client.V1Taint(effect="NoSchedule", key="kwok.x-k8s.io/node", value="fake")
                ]
            ),
            status=client.V1NodeStatus(
                allocatable={
                    "cpu": f"{node.cpu_capacity}m",
                    "memory": f"{node.mem_capacity}Mi",
                    "pods": "50",
                },
                capacity={
                    "cpu": f"{node.cpu_capacity}m",
                    "memory": f"{node.mem_capacity}Mi",
                    "pods": "50",
                },
            ),
        )
        return k8s_node

    def _pod_to_k8s_object(self, pod: Pod, node: Node):
        k8s_pod = client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=client.V1ObjectMeta(name=pod.name),
            spec=client.V1PodSpec(
                containers=[
                    client.V1Container(
                        name=f"{pod.name}-container",
                        image="fake-image",
                        resources=client.V1ResourceRequirements(
                            requests={"cpu": f"{pod.cpu}m", "memory": f"{pod.memory}Mi"},
                            limits={"cpu": f"{pod.cpu}m", "memory": f"{pod.memory}Mi"}
                        )
                    )
                ],
                tolerations=[
                    client.V1Toleration(
                        key="kwok.x-k8s.io/node",
                        operator="Exists",
                        effect="NoSchedule"
                    )
                ],
            )
        )
        if node:
            k8s_pod.spec.node_name = node.name
        
        return k8s_pod
