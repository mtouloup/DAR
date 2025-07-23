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

def main():
    """Run multi-episode training using the DAROTRAIN scheduler."""
    if len(sys.argv) < 2:
        print("Usage: training-controller <config.yaml>")
        sys.exit(1)

    setup_logger(level="INFO", log_file="training.log")
    yaml_files = sys.argv[1:]
    config = load_configs(yaml_files)

    scheduler = None
    controller = Simulator(config)

    episodes = config["training_episodes"]

    for episode in range(episodes):
        logger.info(f"\n=== Starting Episode {episode + 1}/{episodes} ===")

        num_nodes = random.randint(config["training_nodes_per_episode_min"], config["training_nodes_per_episode_max"])
        num_tasks = random.randint(config["training_tasks_per_episode_min"], config["training_tasks_per_episode_max"])

        config["cluster_nodes"] = num_nodes
        cluster = ClusterSynthesizer(config).create_cluster()

        config["workload_tasks"] = num_tasks
        tasks = WorkloadSynthesizer(config).create_tasks()

        if scheduler is None:
            config["cluster_nodes"] = config["training_nodes_per_episode_max"]
            scheduler = SchedulerSelector(config).create_scheduler(cluster)
        else:
            scheduler.onClusterReset(cluster)

        controller.run_simulation(cluster, scheduler, tasks)
        log_rewards(None, None, None, None, mark_end=True)

    logger.info("\n=== Training Completed ===")


if __name__ == "__main__":
    main()
