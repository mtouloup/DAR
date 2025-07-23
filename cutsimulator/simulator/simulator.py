import heapq
import math
from datetime import datetime
import time
from typing import List

from cutsimulator.cluster.cluster import Cluster
from cutsimulator.scheduler.scheduler import Scheduler
from cutsimulator.workload.pod import Pod, PodStatus
from cutsimulator.workload.task import Task
from cutsimulator.evaluation.simulation_statistics import SimulationStatistics  
import logging
logger = logging.getLogger(__name__)


class Simulator:
    def __init__(self, config):
        self.config = config
        self.sim_speedup = self.config['simulation_speedup']
        self.trace = self.config['simulation_save_trace']
        detailedstat = self.config['simulation_detail_statistics']
        self.stats = SimulationStatistics(detailed=detailedstat)

    def append_to_simulation_trace(self, pod, event_type, filename="simulation_trace.txt"):
        with open(filename, 'a') as file:
            file.write(f"{event_type} Event Recorded at {datetime.now()}\n")
            if event_type == "Deployment":
                file.write(f"{pod.name} | {pod.cpu}m | {pod.memory}Mi | {pod.start_time} | {pod.node}\n")
            elif event_type == "Termination":
                file.write(f"{pod.name} | {pod.cpu}m | {pod.memory}Mi | {pod.start_time} -> {pod.end_time} | {pod.duration}s | {pod.node}\n")

    def run_simulation(self, cluster: Cluster, scheduler: Scheduler, tasks: List[Task]):

        self.virtual_time = 0
        self.stats.mark_start(self.virtual_time)
        self.stats.record_cluster_snapshot(cluster.get_nodes())
        self.stats.set_task_count(len(tasks))
        pending_pods = []
        active_pods = []

        for task in tasks:
            for pod in task.get_available_pods():
                pod.status = PodStatus.PENDING
                heapq.heappush(pending_pods, (pod.arrival_time, pod))

        while pending_pods or active_pods:

            # Peek to see the next arrival and finish times (if any)
            next_arrival_time = pending_pods[0][0] if pending_pods else math.inf
            next_finish_time = active_pods[0][0] if active_pods else math.inf

            if next_arrival_time < next_finish_time:
                # Deploy the next pod
                next_arrival_time, pod = heapq.heappop(pending_pods)
                self._simulate_time_passing(next_arrival_time)
                self.stats.record_cluster_utilization(self.virtual_time, cluster.get_nodes())

                node = scheduler.schedule(pod)
                deployed = cluster.deploy_pod(pod, node)

                if deployed:
                    # Pod was successfully deployed
                    pod.status = PodStatus.RUNNING
                    pod.start_time = next_arrival_time
                    pod.end_time = next_arrival_time + pod.duration
                    scheduler.onPodDeployed(pod)
                    heapq.heappush(active_pods, (pod.end_time, pod))
                    self.stats.record_pod_event(pod, success=True)
                    self.append_to_simulation_trace(pod, "Deployment")
                    logger.info(f"Deployed {pod.name} on {pod.node.name} at time {pod.start_time}")
                else:
                    pod.restart_count += 1
                    if pod.restart_count > pod.max_restarts:
                        logger.warning(f"[FAIL] Pod {pod.name} exceeded max restarts - skipping it.")
                        pod.task.unsuccessful = True
                        pod.status = PodStatus.FAILED
                        self.stats.record_pod_event(pod, success=False)
                    elif len(active_pods) == 0:
                        logger.warning(f"[FAIL] Pod {pod.name} does not fit in the cluster - skipping it.")
                        pod.task.unsuccessful = True
                        pod.status = PodStatus.FAILED
                        self.stats.record_pod_event(pod, success=False)
                    else:
                        heapq.heappush(pending_pods, (next_finish_time, pod))
                        logger.info(f"Unable to schedule pod {pod.name} - pushing it back (restart #{pod.restart_count})")
            else:
                # Terminate the next pod
                next_finish_time, pod = heapq.heappop(active_pods)
                self._simulate_time_passing(next_finish_time)
                self.stats.record_cluster_utilization(self.virtual_time, cluster.get_nodes())

                cluster.terminate_pod(pod)
                scheduler.onPodTerminated(pod)
                self.append_to_simulation_trace(pod, "Termination")
                logger.info(f"Terminated pod {pod.name} at time {pod.end_time}")
                pod.status = PodStatus.COMPLETED

                if hasattr(pod, 'task'):
                    pod.task.mark_pod_terminated(pod.name)
                    new_ready = pod.task.get_available_pods()
                    for new_pod in new_ready:
                        if new_pod.node is None and new_pod.status == PodStatus.INITIAL:
                            heapq.heappush(pending_pods, (new_pod.arrival_time, new_pod))
                            new_pod.status = PodStatus.PENDING

        self.stats.mark_end(self.virtual_time)
        self.stats.export_to_csv("simulation_statistics.csv")
        with open("simulation_trace.txt", 'a') as f:
            f.write("\n" + "*" * 50 + " End of Simulation " + "*" * 50 + "\n")
        scheduler.onSimulationEnded()

    def _simulate_time_passing(self, next_time):
        if (next_time < self.virtual_time):
            raise ValueError("Time cannot move backwards!")

        duration = next_time - self.virtual_time
        if duration > 0 and self.sim_speedup > 0:
            time.sleep(duration / self.sim_speedup)

        self.virtual_time = next_time    
