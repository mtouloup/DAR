from cutsimulator.utils.utility import generate_distribution_values
from cutsimulator.workload.pod import Pod
from cutsimulator.workload.task import Task


class WorkloadSynthesizer:
    def __init__(self, config):
        required_keys = [
            'workload_tasks',
            'workload_pods_number_dist',
            'workload_pods_cpu_dist',
            'workload_pods_mem_dist',
            'workload_pods_interarrival_dist',
            'workload_pods_duration_dist',
            'workload_pods_max_restarts'
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing workload config key: {key}")
        self.config = config

    # If we want pod-centric simulations, use this function
    def create_pods(self):  # → returns List[Pod]
        num_pods = self.config['workload_tasks'] # Assuming 1 pod per task for this specific case
        cpu_dist = self.config['workload_pods_cpu_dist']
        mem_dist = self.config['workload_pods_mem_dist']
        interarrival_dist = self.config['workload_pods_interarrival_dist']
        duration_dist = self.config['workload_pods_duration_dist']

        cpus = generate_distribution_values(cpu_dist, num_pods)
        memories = generate_distribution_values(mem_dist, num_pods)
        interarrivals = generate_distribution_values(interarrival_dist, num_pods)
        durations = generate_distribution_values(duration_dist, num_pods)

        pods = []
        arrival_time = 0
        for i in range(num_pods):
            pods.append(Pod(f"pod-{i + 1}", cpus[i], memories[i], durations[i], arrival_time + interarrivals[i]))
            arrival_time += interarrivals[i]

        return pods
    
    # For task-centric simulation   
    def create_tasks(self):  # → returns List[Task]
        num_tasks = self.config['workload_tasks']
        pod_count_dist = self.config['workload_pods_number_dist']  # Now a distribution
        interarrival_dist = self.config['workload_pods_interarrival_dist']
        
        pod_cpu_dist = self.config['workload_pods_cpu_dist']
        pod_mem_dist = self.config['workload_pods_mem_dist']
        pod_duration_dist = self.config['workload_pods_duration_dist']
        pod_max_restarts = self.config['workload_pods_max_restarts']

        interarrivals = generate_distribution_values(interarrival_dist, num_tasks)
        pod_counts = generate_distribution_values(pod_count_dist, num_tasks)

        tasks = []
        arrival_time = 0

        for i in range(num_tasks):
            task_name = f"task-{i+1}"
            arrival_time += interarrivals[i]
            num_pods = int(pod_counts[i])  # Ensure it's an integer
            
            task_args = {
                "pods_cpu_dist": pod_cpu_dist,
                "pods_mem_dist": pod_mem_dist,
                "pods_duration_dist": pod_duration_dist,
                "max_restarts": pod_max_restarts
            }
            
            task = Task(
                name=task_name,
                num_pods=num_pods,
                pod_config=task_args,
                arrival_time=arrival_time
            )
            tasks.append(task)

        return tasks


