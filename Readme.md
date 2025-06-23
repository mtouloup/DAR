
# Simulator Usage

## Prerequisites

- [KWOK](https://kwok.sigs.k8s.io/docs/user/installation/) must be installed and configured for KWOK-based simulation.
- `kubectl` must be installed and configured for Kubernetes access.
- Python 3.8+ environment with required packages (`PyYAML`, `kubernetes`, etc.)

---

## Simulator Overview

This simulator enables training and evaluation of scheduling strategies (including **multi-agent reinforcement learning**) in a Kubernetes-like cluster environment.

Supports:

- **KWOKCluster**: Real Kubernetes pod simulation via KWOK
- **PythonCluster**: Lightweight Python-only emulation
- **Schedulers**: 
  - `DEFAULT`: Kubernetes Native Scheduler
  - `DARO`: Decentralized Agent-based Reinforcement Optimizer using QMIX
  - `RoundRobin`: RoundRobin assignment of pods

---

## Step 1: Define Your Cluster

Define your cluster characteristics in `cluster_descriptor.yaml`:

```yaml
Cluster:
  Type: KWOK  # KWOK or Python
  Reset: True    # True: Clear existing cluster, False: Append to existing cluster
  Nodes:
    Number: 10    # Number of nodes
    CPU Dist: {type: poisson, mean: 5000, min: 1000, max: 8000, round: -2} # in millicores
    Mem Dist: {type: normal, mean: 6000, stdev: 2000, min: 2000, max: 8000, round: -1} # in Mi
 ```

## Step 2: Define Your Workload

Define your workload characteristics in `workload_descriptor.yaml`:

```yaml
Workload:
  Pods:
    Number: 6
    CPU Dist: {type: normal, mean: 2000, stdev: 500, min: 500, max: 4000, round: -2} # in millicores
    Mem Dist: {type: normal, mean: 3000, stdev: 2000, min: 1000, max: 5000, round: -1} # in Mi
    Interarrival Dist: {type: poisson, mean: 5, min: 3, max: 6} # in seconds
    Duration Dist: {type: poisson, mean: 4, min: 2, max: 9} # in seconds
```

## Step 3: Configure Scheduler Configurations

Define your scheduler characteristics in `scheduler_descriptor.yaml`:

```yaml
Scheduler:
  Type: DARO  # DARO or ROUNDROBIN or DEFAULT
  Params:      # Apply only to DARO
    output_dims: 11     # actions/bids
    LearningRate: 0.001
    GAMMA: 0.99
    Epsilon: 0.1
    Replay_buffer_size: 1000
    BatchSize: 32
  Reward:
    Type: LB_reward  # Load Balancing reward
  SimSpeedup: 1  # 1=real-time, 0=infinite, other numbers=speedup factor
```

## Step 4: Configure Training Configurations

Update `training_config.yaml`:

```yaml
Training:
  Episodes: 10            # Number of training episodes
  Nodes: [4, 6]           # Range: number of nodes (min, max)
  Pods: [10, 20]          # Range: number of pods (min, max)
```

---

## Step 5: Run Training

To start multi-episode training:

```bash
python3 training_controller.py cluster_descriptor.yaml workload_descriptor.yaml scheduler_descriptor.yaml training_descriptor.yaml
```

This will:
- Loads all descriptor files (order does not matter)
- Randomizes number of nodes/pods for each episode
- Resets the cluster before each episode
- Applies the scheduler (`DARO`, `ROUNDROBIN`, or `DEFAULT`)
- Trains using QMIX (only if using `DARO` scheduler)
- Saves model as `qmix_trained_model.pth`

---

## Optional: Run a Single Simulation (Standalone Mode)

You can simulate a one-time workload outside of training:

```bash
python3 simulation_controller.py cluster_descriptor.yaml workload_descriptor.yaml scheduler_descriptor.yaml
```

This is useful for testing or visualizing specific workload behavior.

---

## Output Files

- `simulation_trace.txt`: Contains deployment and termination logs of each pod.
- `cluster_info.txt` *(optional)*: Can be generated to record cluster specs.
- `qmix_trained_model.pth`: Trained QMIX agent model (only for DARO scheduler).
- `reward_trace.csv`: Contains rewards assigned to each node in every episode.

---

## Supported Distribution Formats

You can define distributions for **CPU**, **Memory**, **Interarrival**, and **Duration**:

| Type     | Format Example |
|----------|----------------|
| Normal   | `{type: normal, mean: 6, stdev: 2, min: 2, max: 8, round: 1}` |
| Poisson  | `{type: poisson, mean: 6, min: 2, max: 8, round: 1}` |
| Uniform  | `{type: uniform, min: 2, max: 8, round: 1}` |
| Fixed    | `{type: fixed, value: 4}` |

> `round` (optional): Rounds output to given decimal.  
> Units:
> - CPU: millicores
> - Memory: Mi
> - Time (Interarrival/Duration): seconds  
> Kubernetes **expects integer memory values for pods**.


## Contact

For questions or contributions, please contact the CUT team.

