import numpy as np
import csv
import os
import yaml
from pathlib import Path

def generate_distribution_values(distribution, count):
    if distribution['type'] not in {"normal", "poisson", "uniform", "fixed"}:
        raise ValueError(f"Unsupported distribution {distribution['type']}")

    if distribution['type'] == 'fixed':
        # Generate fixed values
        if 'value' not in distribution:
            raise ValueError("Missing value in 'fixed' distribution")
        
        return np.full(count, distribution['value'])

    # Other distributions have min and max values
    min = distribution['min']
    max = distribution['max']
    if min >= max:
        raise ValueError("Min has to be smaller than max")

    if 'mean' in distribution:
        if distribution['mean'] < min or distribution['mean'] > max:
            raise ValueError("Mean has to be between min and max")

    roundVal = 0
    if 'round' in distribution:
        roundVal = distribution['round']
        
    # Generate distributed numbers within the range
    sequence = []
    while len(sequence) < count:
        
        if distribution['type'] == 'normal':
            sample = np.random.normal(loc=distribution['mean'], scale=distribution['stdev'])
        elif distribution['type'] == 'poisson':
            sample = np.random.poisson(lam=distribution['mean'])
        elif distribution['type'] == 'uniform':
            sample = np.random.uniform(low=distribution['min'], high=distribution['max'])
        
        if min <= sample <= max:
            sequence.append(int(round(sample, roundVal)))           
    return sequence

def convert_cpu(cpu: str) -> int:
    # CPU conversion (convert to millicores)
    if cpu.endswith("m"):
        cpu_value = int(cpu[:-1])  # Already in millicores
    else:
        cpu_value = round(float(cpu) * 1000)  # Convert cores to millicores

    return cpu_value

def convert_memory(memory: str) -> int:
    # Memory conversion (convert to MiB)
    if memory.endswith("Ki"):
        memory_value = round(float(memory[:-2]) // 1024)  # Convert Ki to Mi
    elif memory.endswith("Mi"):
        memory_value = int(memory[:-2])  # Already in Mi
    elif memory.endswith("Gi"):
        memory_value = round(float(memory[:-2]) * 1024)  # Convert Gi to Mi
    else:
        memory_value = round(float(memory) // (1024*1024))  # Convert Bytes to Mi

    return memory_value

def safe_ratio(numerator, denominator, default_if_zero=0):
    return numerator / denominator if denominator > 0 else default_if_zero
    
def load_configs(yaml_files):
    config = {}
    yaml_paths = [Path(f) for f in yaml_files]
    for path in yaml_paths:
        with open(path, 'r') as f:
            new_config = yaml.safe_load(f) or {}  # Avoid issues if file is empty
            config = _deep_merge_dicts(config, new_config)
    return config

def _deep_merge_dicts(dict1, dict2):
    """Recursively merge dict2 into dict1."""
    for key in dict2:
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            _deep_merge_dicts(dict1[key], dict2[key])
        else:
            dict1[key] = dict2[key]
    return dict1

def log_rewards(pod_name, selected_node, valid_nodes, rewards, mark_end=False, log_file="reward_trace.csv"):
    is_new = not os.path.exists(log_file)

    with open(log_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if is_new:
            writer.writerow(["Pod", "Node", "Is_Selected", "Reward"])

        if mark_end:
            writer.writerow(["--- END EPISODE ---", "", "", ""])
            return

        for node, reward in zip(valid_nodes, rewards):
            writer.writerow([
                pod_name,
                node.name,
                int(node == selected_node),
                round(reward, 4)
            ])