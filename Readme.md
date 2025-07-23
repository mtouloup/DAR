# K8s Workload Simulator

A modular and extensible simulator for evaluating diverse pod scheduling strategies in Kubernetes-like environments. Supports customizable clusters, workloads, and schedulers — including rule-based and learning-based approaches.


---

##  Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

- For **KWOK-based clusters**:
  - Install [KWOK](https://kwok.sigs.k8s.io/docs/user/installation/)
  - Ensure `kubectl` is configured

---

## Standalone Simulation

To run a one-time simulation:

```bash
python3 scripts/simulation_controller.py configs/config.yaml
```

This uses our YAML configuration (`/configs/config.yaml`) to:
- Deploy a synthetic cluster (KWOK or Python)
- Generate pod workloads based on task/pod distributions
- Apply the selected scheduler (e.g., `ROUNDROBIN`, `DEFAULT`, or `DAROTRAIN`)
- Save simulation traces and rewards (if enabled)


**YAML Parameters for Simulation** include:

**🔹 Cluster Parameters**
- `cluster_type`, `cluster_reset`
- `cluster_nodes`, `cluster_nodes_cpu_dist`, `cluster_nodes_mem_dist`

**🔹 Workload Parameters**
- `workload_tasks`
- `workload_pods_number_dist`, `workload_pods_cpu_dist`, `workload_pods_mem_dist`
- `workload_pods_interarrival_dist`, `workload_pods_duration_dist`, `workload_pods_max_restarts`

**🔹 Scheduler Parameters**
- `scheduler_type`

**🔹 Simulation Settings**
- `simulation_speedup`
- `simulation_save_trace`, `simulation_detail_statistics`

**🔹 Training Parameters**
- `training_episodes`
- `training_nodes_per_episode_min`, `training_nodes_per_episode_max`
- `training_tasks_per_episode_min`, `training_tasks_per_episode_max`
---

##  Multi-Episode Training

To launch MARL-based training using the **DAROTRAIN** scheduler:

```bash
python3 scripts/training_controller.py configs/config.yaml
```

The training process will:
- Randomize cluster size and workload per episode
- Schedule pods using the DAROTRAIN (QMIX) agent
- Train and update the agent using reward feedback
- Save model weights (`qmix_latest.pth`) and logs

**Additional YAML Parameters for Training**:
- All `scheduler_daro_*` hyperparameters (learning rate, gamma, etc.)

---

##  Output Artifacts

| File                   | Description                                 |
|------------------------|---------------------------------------------|
| `simulation_trace.txt` | Deployment and termination events           |
| `reward_trace.csv`     | Per-node reward values for each pod         |
| `qmix_latest.pth`      | Trained QMIX model (only for DAROTRAIN)     |
| `cluster_info.txt`     | Final cluster specification snapshot        |

---

## Configurable Components

All settings are defined in a **single flattened YAML** (`configs/config.yaml`):

- Cluster type, size, and node resource distributions
- Workload task structure and pod arrival/duration/resource distributions
- Scheduler type and parameters (including DAROTRAIN hyperparameters)
- Simulation toggles and speed
- Training episode counts and node/task ranges

---

##  Supported Schedulers

| Scheduler     | Description                                     |
|---------------|-------------------------------------------------|
| `DEFAULT`     | Native Kubernetes (KWOK) scheduler               |
| `ROUNDROBIN`  | Simple round-robin node selection               |
| `DAROTRAIN`   | Decentralized RL scheduler using QMIX           |

---

##  Supported Distributions

You can configure the following statistical distributions:
- `fixed`, `normal`, `poisson`, `uniform`
- Fields: CPU, memory, pod interarrival, duration, number of pods per task

| Type     | Format Example |
|----------|----------------|
| Normal   | `{type: normal, mean: 6, stdev: 2, min: 2, max: 8, round: 1}` |
| Poisson  | `{type: poisson, mean: 6, min: 2, max: 8, round: 1}` |
| Uniform  | `{type: uniform, min: 2, max: 8, round: 1}` |
| Fixed    | `{type: fixed, value: 4}` |

Units:
- CPU: millicores
- Memory: Mi (Kubernetes expects integer memory values for pods)
- Time (Interarrival/Duration): seconds  

`round` (optional): Rounds output to given decimal.  

---

##  Contact

Developed and maintained by the **CUT**.  
For issues or contributions, please contact us or submit a pull request.

