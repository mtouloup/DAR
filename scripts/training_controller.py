import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutsimulator.cluster.cluster_synthesizer import ClusterSynthesizer
from cutsimulator.scheduler.scheduler_selector import SchedulerSelector
from cutsimulator.workload.workload_synthesizer import WorkloadSynthesizer
from cutsimulator.simulator.simulator import Simulator
from cutsimulator.utils.utility import log_rewards, load_configs, setup_logger
import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Load the training descriptor
    import sys
    if len(sys.argv) < 2:
        # Order of yaml files does not matter
        print("Usage: python3 training_controller.py configs.yaml")
        sys.exit(1)
    
    setup_logger(level="INFO", log_file="training.log")
    yaml_files = sys.argv[1:]
    config = load_configs(yaml_files)

    scheduler = None
    controller = Simulator(config)

    episodes = config["training_episodes"]

    for episode in range(episodes):
        logger.info(f"\n=== Starting Episode {episode + 1}/{episodes} ===")

        # Randomize number of nodes and tasks within specified ranges
        num_nodes = random.randint(config["training_nodes_per_episode_min"], config["training_nodes_per_episode_max"])
        num_tasks = random.randint(config["training_tasks_per_episode_min"], config["training_tasks_per_episode_max"])

        # Update cluster
        config["cluster_nodes"] = num_nodes
        cluster = ClusterSynthesizer(config).create_cluster()

        # Update workload
        config["workload_tasks"] = num_tasks
        tasks = WorkloadSynthesizer(config).create_tasks()

        # Update scheduler
        if scheduler is None:
            # Setting the number of nodes to max to correctly create QMIX networks
            config["cluster_nodes"] = config["training_nodes_per_episode_max"]
            scheduler = SchedulerSelector(config).create_scheduler(cluster)
        else:
            scheduler.onClusterReset(cluster)

        # Run a single simulation (stats and export are handled internally)
        controller.run_simulation(cluster, scheduler, tasks)

        # Log RL reward separation per episode
        log_rewards(None, None, None, None, mark_end=True)

    logger.info("\n=== Training Completed ===")
