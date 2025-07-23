import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutsimulator.cluster.cluster_synthesizer import ClusterSynthesizer

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 cluster_controller.py <config.yaml>")
        sys.exit(1)

    # Load the cluster descriptor YAML file.
    file_path = sys.argv[1]
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    # Use the synthesizer to create the cluster
    synthesizer = ClusterSynthesizer(config)
    synthesizer.create_cluster()
