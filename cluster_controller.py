import yaml
import sys
from cluster.cluster_synthesizer import ClusterSynthesizer

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 cluster_controller.py <cluster_descriptor.yaml>")
        sys.exit(1)

    # Load the cluster descriptor YAML file.
    file_path = sys.argv[1]
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    # Use the synthesizer to create the cluster
    synthesizer = ClusterSynthesizer(config)
    synthesizer.create_cluster()
