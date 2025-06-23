import random
from cluster.cluster_synthesizer import ClusterSynthesizer
from scheduler.scheduler_selector import SchedulerSelector
from workload.workload_synthesizer import WorkloadSynthesizer
from simulation_controller import SimulationController
from utils.utility import log_rewards, load_configs

if __name__ == "__main__":
    # Load the training descriptor
    import sys
    if len(sys.argv) < 2:
        # Order of yaml files does not matter
        print("Usage: python3 training_controller.py cluster_descriptor.yaml workload_descriptor.yaml scheduler_descriptor.yam training_descriptor.yaml")
        sys.exit(1)

    yaml_files = sys.argv[1:]
    config = load_configs(yaml_files)

    scheduler = None
    controller = SimulationController(config)

    episodes = config["Training"]["Episodes"]

    for episode in range(episodes):
        print(f"\n=== Starting Episode {episode + 1}/{episodes} ===")

        # Randomize number of nodes and pods within specified ranges
        num_nodes = random.randint(config["Training"]["Nodes"][0], config["Training"]["Nodes"][1])
        num_pods = random.randint(config["Training"]["Pods"][0], config["Training"]["Pods"][1])

        # Update cluster
        config["Cluster"]["Nodes"]["Number"] = num_nodes
        cluster = ClusterSynthesizer(config).create_cluster()

        # Update workload
        config["Workload"]["Pods"]["Number"] = num_pods
        pods = WorkloadSynthesizer(config).create_pods()

        # Update scheduler
        if scheduler is None:
            # Setting the number of nodes to max to correctly create QMIX networks
            config["Cluster"]["Nodes"]["Number"] = config["Training"]["Nodes"][1]
            scheduler = SchedulerSelector(config).create_scheduler(cluster)
        else:
            scheduler.onClusterReset(cluster)

        # Run a single simulation
        controller.run_simulation(cluster, scheduler, pods)
        log_rewards(None, None, None, None, mark_end=True)

    print("\n=== Training Completed ===")
