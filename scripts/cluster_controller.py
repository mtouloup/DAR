import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutsimulator.cluster.cluster_synthesizer import ClusterSynthesizer

def main():
    """Create a cluster based on a YAML configuration file."""
    if len(sys.argv) != 2:
        print("Usage: cluster-controller <config.yaml>")
        sys.exit(1)

    file_path = sys.argv[1]
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    synthesizer = ClusterSynthesizer(config)
    synthesizer.create_cluster()


if __name__ == "__main__":
    main()
