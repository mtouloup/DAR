import heapq
import math
from datetime import datetime, timedelta
import time
from typing import List
from cluster.cluster import Cluster
from cluster.cluster_synthesizer import ClusterSynthesizer
from scheduler.scheduler import Scheduler
from scheduler.scheduler_selector import SchedulerSelector 
from workload.pod import Pod
from workload.workload_synthesizer import WorkloadSynthesizer
from utils.utility import load_configs

     
class SimulationController:    
    def __init__(self, config):
        self.config = config
        self.sim_speedup = self.config['Scheduler']['SimSpeedup']

    def append_to_simulation_trace(self, pod, event_type, filename="simulation_trace.txt"):
        with open(filename, 'a') as file:
            file.write("\n" + "*" * 50 + "\n")
            file.write(f"{event_type} Event Recorded at {datetime.now()}\n")
            file.write("*" * 50 + "\n")
            if event_type == "Deployment":
                file.write(f"{pod.name} | {pod.cpu}m | {pod.memory}Mi | {pod.start_time} | {pod.node}\n")
            elif event_type == "Termination":
                file.write(f"{pod.name} | {pod.cpu}m | {pod.memory}Mi | {pod.start_time} -> {pod.end_time} | {pod.duration}s | {pod.node}\n")

    def run_simulation(self, cluster : Cluster, scheduler : Scheduler, pods : List[Pod]):

        self.virtual_time = 0
        pending_pods = []
        active_pods = []

        for pod in pods:
            heapq.heappush(pending_pods, (pod.arrival_time, pod))

        while pending_pods or active_pods:

            # Peek to see the next arrival and finish times (if any)
            next_arrival_time = pending_pods[0][0] if pending_pods else math.inf
            next_finish_time = active_pods[0][0] if active_pods else math.inf

            if next_arrival_time < next_finish_time:
                # Deploy the next pod
                next_arrival_time, pod = heapq.heappop(pending_pods)
                self._simulate_time_passing(next_arrival_time)

                node = scheduler.schedule(pod)
                deployed = cluster.deploy_pod(pod, node)

                if deployed:
                    # Pod was successfully deployed
                    pod.start_time = next_arrival_time
                    pod.end_time = next_arrival_time + pod.duration
                    scheduler.onPodDeployed(pod)
                    heapq.heappush(active_pods, (pod.end_time, pod))
                    self.append_to_simulation_trace(pod, "Deployment")
                    print(f"Deployed {pod.name} on {pod.node.name} at time {pod.start_time}")
                else:
                    # Pod was not deployed
                    if active_pods:
                        # Push it back to the pending pods, after the next finish time
                        heapq.heappush(pending_pods, (next_finish_time, pod))
                        print(f"Unable to schedule pod {pod} - pushing it back")
                    else:
                        print(f"Unable to schedule pod {pod} - skipping it")
            else:
                # Terminate the next pod
                next_finish_time, pod = heapq.heappop(active_pods)
                self._simulate_time_passing(next_finish_time)
                cluster.terminate_pod(pod)
                scheduler.onPodTerminated(pod)
                self.append_to_simulation_trace(pod, "Termination")
                print(f"Terminated pod {pod.name} at time {pod.end_time}")

        # Notify the scheduler that the simulation is done
        scheduler.onSimulationEnded()


    def _simulate_time_passing(self, next_time):
        if (next_time < self.virtual_time):
            raise ValueError("Time cannot move backwards!")

        duration = next_time - self.virtual_time
        if duration > 0 and self.sim_speedup > 0:
            time.sleep(duration / self.sim_speedup)

        self.virtual_time = next_time    


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        # Order of yaml files does not matter
        print("Usage: python3 simulation_controller.py cluster_descriptor.yaml workload_descriptor.yaml scheduler_descriptor.yaml")
        sys.exit(1)

    yaml_files = sys.argv[1:]
    config = load_configs(yaml_files)

    cluster = ClusterSynthesizer(config).create_cluster()
    scheduler = SchedulerSelector(config).create_scheduler(cluster)
    pods = WorkloadSynthesizer(config).create_pods()

    controller = SimulationController(config)
    controller.run_simulation(cluster, scheduler, pods)
