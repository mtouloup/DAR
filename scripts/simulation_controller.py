import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutsimulator.cluster.cluster_synthesizer import ClusterSynthesizer
from cutsimulator.scheduler.scheduler_selector import SchedulerSelector
from cutsimulator.workload.workload_synthesizer import WorkloadSynthesizer
from cutsimulator.utils.utility import load_configs, setup_logger
from cutsimulator.simulator.simulator import Simulator

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 simulator.py config.yaml")
        sys.exit(1)
    
    # Setup logging system
    setup_logger(level="INFO", log_file="simulator.log")
    yaml_files = sys.argv[1:]
    config = load_configs(yaml_files)

    cluster = ClusterSynthesizer(config).create_cluster()
    scheduler = SchedulerSelector(config).create_scheduler(cluster)
    tasks = WorkloadSynthesizer(config).create_tasks()

    simulator = Simulator(config)
    simulator.run_simulation(cluster, scheduler, tasks)
