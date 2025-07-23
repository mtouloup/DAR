import numpy as np

from cutsimulator.workload.pod import Pod, PodStatus
from cutsimulator.utils.utility import generate_distribution_values

class Task:
    def __init__(self, name, num_pods, pod_config, arrival_time):
        self.name = name
        self.arrival_time = arrival_time
        self.length = num_pods
        self.unsuccessful = False  # Becomes True if any pod exceeds restart limit

        # Generate pods for this task
        self.pods = self._generate_pods(pod_config)
        
        # Create a DAG dependency matrix (lower triangular = backward dependencies)
        self.dag = np.tril(np.random.randint(2, size=(self.length, self.length)), k=-1)

        self.available_pods = {}
        self.terminated = False
        self.update_available_pods()

        # Store pod key order for consistent indexing
        self.pod_keys = list(self.pods.keys())

        # Back-reference each pod to this Task
        for pod in self.pods.values():
            pod.task = self

    def _generate_pods(self, pod_config):
        cpu_dist = pod_config['pods_cpu_dist']
        mem_dist = pod_config['pods_mem_dist']
        duration_dist = pod_config['pods_duration_dist']
        max_restarts = pod_config['max_restarts']

        cpus = generate_distribution_values(cpu_dist, self.length)
        mems = generate_distribution_values(mem_dist, self.length)
        durations = generate_distribution_values(duration_dist, self.length)

        pods = {}
        for i in range(self.length):
            pod_name = f"{self.name}-pod-{i}"
            pods[pod_name] = Pod(
                name=pod_name,
                cpu=int(cpus[i]),
                memory=int(mems[i]),
                duration=int(durations[i]),
                arrival_time=self.arrival_time,  # same for all pods in this task
                max_restarts=max_restarts
            )
        return pods

    def update_available_pods(self):
        # Updates the set of pods ready for deployment based on DAG and termination status.     
        self.available_pods.clear()
        prev_keys = []

        if self.unsuccessful:
            return

        for i, k in enumerate(self.pods.keys()):
            pod = self.pods[k]

            if pod.status in [PodStatus.RUNNING, PodStatus.COMPLETED, PodStatus.FAILED]:
                prev_keys.append(k)
                continue

            available = True
            dependency_end_times = []

            for j, prev_k in enumerate(prev_keys):
                if self.dag[i, j] == 1:
                    parent = self.pods[prev_k]
                    if parent.status != PodStatus.COMPLETED:
                        available = False
                        break
                    else:
                        dependency_end_times.append(parent.end_time)

            if available:
                # Update arrival_time if dependencies exist
                if dependency_end_times:
                    pod.arrival_time = max(dependency_end_times)
                self.available_pods[k] = pod

            prev_keys.append(k)

        self.terminated = all(p.status == PodStatus.COMPLETED for p in self.pods.values())


    def mark_pod_terminated(self, pod_name):
        if pod_name in self.pods and self.pods[pod_name].status != PodStatus.COMPLETED:
            self.pods[pod_name].status = PodStatus.COMPLETED
            self.update_available_pods()

    def get_available_pods(self):  # Returns list of unstarted pods
        return list(self.available_pods.values())

    def __repr__(self):
        return f"Task(name={self.name}, pods={[p.name for p in self.pods.values()]})"
    
    def is_successful(self):
        return not self.unsuccessful and self.terminated