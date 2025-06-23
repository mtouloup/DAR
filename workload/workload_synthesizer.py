from utils.utility import generate_distribution_values
from workload.pod import Pod

class WorkloadSynthesizer:
    def __init__(self, config):
        if 'Workload' not in config:
            raise ValueError("WorkloadSynthesizer requires the 'Workload' configuration")
        self.config = config

    def create_pods(self):
        num_pods = self.config['Workload']['Pods']['Number']
        cpu_dist = self.config['Workload']['Pods']['CPU Dist']
        mem_dist = self.config['Workload']['Pods']['Mem Dist']
        interarrival_dist = self.config['Workload']['Pods']['Interarrival Dist']
        duration_dist = self.config['Workload']['Pods']['Duration Dist']

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
