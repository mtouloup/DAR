import csv
import numpy as np

from cutsimulator.utils.utility import log_statistics

class LoadBalancingStatus:
    def __init__(self):
        self.trace = {}  # time -> {"cpu_std": val, "mem_std": val, "avg_cpu": val, "avg_mem": val}

    def record(self, timestamp, nodes):
        cpu_usages = [1 - node.cpu_available / node.cpu_capacity if node.cpu_capacity else 0 for node in nodes]
        mem_usages = [1 - node.mem_available / node.mem_capacity if node.mem_capacity else 0 for node in nodes]
        self.trace[timestamp] = {
            "cpu_std": np.std(cpu_usages),
            "mem_std": np.std(mem_usages),
            "avg_cpu": np.mean(cpu_usages),
            "avg_mem": np.mean(mem_usages),
            "min_cpu": np.min(cpu_usages),
            "max_cpu": np.max(cpu_usages),
            "min_mem": np.min(mem_usages),
            "max_mem": np.max(mem_usages)
        }

    def aggregate(self):
        if not self.trace:
            return {
                "cpu_std": 0, "mem_std": 0,
                "avg_cpu": 0, "avg_mem": 0,
                "min_cpu": 0, "max_cpu": 0,
                "min_mem": 0, "max_mem": 0
            }

        cpu_stds = []
        mem_stds = []
        avg_cpus = []
        avg_mems = []
        min_cpus = []
        max_cpus = []
        min_mems = []
        max_mems = []

        for entry in self.trace.values():
            cpu_stds.append(entry["cpu_std"])
            mem_stds.append(entry["mem_std"])
            avg_cpus.append(entry["avg_cpu"])
            avg_mems.append(entry["avg_mem"])
            min_cpus.append(entry["min_cpu"])
            max_cpus.append(entry["max_cpu"])
            min_mems.append(entry["min_mem"])
            max_mems.append(entry["max_mem"])

        return {
            "cpu_std": float(np.mean(cpu_stds)),
            "mem_std": float(np.mean(mem_stds)),
            "avg_cpu": float(np.mean(avg_cpus)),
            "avg_mem": float(np.mean(avg_mems)),
            "min_cpu": float(np.min(min_cpus)),
            "max_cpu": float(np.max(max_cpus)),
            "min_mem": float(np.min(min_mems)),
            "max_mem": float(np.max(max_mems))
        }

class SimulationStatistics:
    def __init__(self, detailed=False):
        self.pod_stats = []  # Holds dicts with pod lifecycle info
        self.load_balancer = LoadBalancingStatus()
        self.simulation_start = None
        self.simulation_end = None
        self.detailed = detailed
        self.cluster_nodes = []  # Persist final snapshot for capacity info
        self.num_tasks = 0

    def mark_start(self, timestamp):
        self.simulation_start = timestamp

    def mark_end(self, timestamp):
        self.simulation_end = timestamp

    def record_cluster_snapshot(self, nodes):
        self.cluster_nodes = nodes  # save for post-analysis

    def set_task_count(self, num_tasks):
        self.num_tasks = num_tasks

    def record_pod_event(self, pod, success=True):
        self.pod_stats.append({
            "pod": pod.name,
            "arrival": pod.arrival_time,
            "start": pod.start_time,
            "end": pod.end_time,
            "success": success
        })

    def record_cluster_utilization(self, timestamp, nodes):
        self.load_balancer.record(timestamp, nodes)

    def compute_final_metrics(self):
        completed = [p for p in self.pod_stats if p['success'] and p['end'] is not None]
        wait_times = [p['start'] - p['arrival'] for p in completed if p['start'] is not None and p['start'] > p['arrival']]
        latencies = [p['end'] - p['arrival'] for p in completed if p['end'] is not None]

        duration = self.simulation_end - self.simulation_start if self.simulation_start is not None and self.simulation_end is not None else 0
        throughput = len(completed) / duration if duration > 0 else 0
        rejection_rate = 1 - len(completed) / len(self.pod_stats) if self.pod_stats else 0
        makespan = max([p['end'] for p in completed], default=0) - min([p['arrival'] for p in completed], default=0)

        lb_agg = self.load_balancer.aggregate()

        cpu_caps = [n.cpu_capacity for n in self.cluster_nodes]
        mem_caps = [n.mem_capacity for n in self.cluster_nodes]

        return {
            "total_pods": len(self.pod_stats),
            "completed_pods": len(completed),
            "rejection_rate": round(rejection_rate, 4),
            "avg_wait_time": round(np.mean(wait_times), 4) if wait_times else 0,
            "min_wait_time": round(np.min(wait_times), 4) if wait_times else 0,
            "max_wait_time": round(np.max(wait_times), 4) if wait_times else 0,
            "avg_latency": round(np.mean(latencies), 4) if latencies else 0,
            "min_latency": round(np.min(latencies), 4) if latencies else 0,
            "max_latency": round(np.max(latencies), 4) if latencies else 0,
            "throughput": round(throughput, 4),
            "makespan": round(makespan, 4),
            "num_nodes": len(self.cluster_nodes),
            "min_cpu_capacity": min(cpu_caps) if cpu_caps else 0,
            "max_cpu_capacity": max(cpu_caps) if cpu_caps else 0,
            "min_mem_capacity": min(mem_caps) if mem_caps else 0,
            "max_mem_capacity": max(mem_caps) if mem_caps else 0,
            "avg_cpu_std": round(lb_agg["cpu_std"], 4),
            "avg_mem_std": round(lb_agg["mem_std"], 4),
            "avg_cpu_util": round(lb_agg["avg_cpu"], 4),
            "avg_mem_util": round(lb_agg["avg_mem"], 4),
            "min_cpu_util": round(lb_agg["min_cpu"], 4),
            "max_cpu_util": round(lb_agg["max_cpu"], 4),
            "min_mem_util": round(lb_agg["min_mem"], 4),
            "max_mem_util": round(lb_agg["max_mem"], 4),
            "total_tasks": self.num_tasks
        }

    def export_to_csv(self, path):
        metrics = self.compute_final_metrics()

        # Reorder keys: Nodes → Tasks → Pods → others
        reordered = {}
        for key in ["num_nodes", "total_tasks", "total_pods", "completed_pods"]:
            if key in metrics:
                reordered[key] = metrics.pop(key)
        reordered.update(metrics)

        # Write main summary CSV
        log_statistics(reordered, path)

        # Write detailed trace if enabled
        if self.detailed:
            detailed_path = path.replace(".csv", "_detailed.csv")
            with open(detailed_path, mode='w', newline='') as f:
                fieldnames = ["timestamp", "num_nodes", "total_tasks", "total_pods"] + list(self.load_balancer.trace[list(self.load_balancer.trace.keys())[0]].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for t, row in self.load_balancer.trace.items():
                    writer.writerow({
                        "timestamp": t,
                        "num_nodes": len(self.cluster_nodes),
                        "total_tasks": self.num_tasks,
                        "total_pods": len(self.pod_stats),
                        **row
                    })

    def reset(self):
        self.__init__(self.detailed)