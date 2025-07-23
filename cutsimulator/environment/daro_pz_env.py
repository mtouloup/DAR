import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import threading
from typing import List
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces

import cutsimulator.state.obs_builder as ob
from cutsimulator.environment.coordinator import Coordinator
from cutsimulator.environment.daro_pz_scheduler import DaroPettingZooScheduler
from cutsimulator.cluster.cluster import Cluster
from cutsimulator.cluster.cluster_synthesizer import ClusterSynthesizer
from cutsimulator.scheduler.scheduler import Scheduler
from cutsimulator.simulator.simulator import Simulator
from cutsimulator.workload.task import Task
from cutsimulator.workload.workload_synthesizer import WorkloadSynthesizer

# A helper thread for running the simulation
class SimulatorThread(threading.Thread):
    def __init__(self, controller: Simulator, 
                 cluster : Cluster, scheduler : Scheduler, tasks : List[Task]):
        super().__init__()
        self.controller = controller
        self.cluster = cluster
        self.scheduler = scheduler
        self.tasks = tasks

    def run(self):
        self.controller.run_simulation(self.cluster, self.scheduler, self.tasks)

# A PettingZoo environment for the DARO framework
class DaroPettingZooEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "daro_env_v0"}

    def __init__(self, config: dict):
        """
        Initializes the environment
        """
        super().__init__()
        self.config = config
        
        # Create key parameters
        max_num_agents = config["training_nodes_per_episode_max"] # equals max num of nodes
        self.possible_agents = [f"agent_{i}" for i in range(max_num_agents)]
        self.agents = self.possible_agents.copy()
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(ob.obs_dimensions(),), 
                                            dtype=np.float32)
        self._action_space = spaces.Discrete(11)

        self._full_action_space_mask = np.ones(self._action_space.n, dtype=bool)
        self._full_action_space_mask[0] = False
        self._zero_action_space_mask = np.zeros(self._action_space.n, dtype=bool)
        self._zero_action_space_mask[0] = True

        # Initialize key simulation-related classes
        self.coordinator = Coordinator(main_turn_first=False)
        self.scheduler = DaroPettingZooScheduler(config, self.coordinator)
        self.sim_controller = Simulator(config)
        self.sim_thread = None

    def reset(self, seed=None, options=None):
        """
        Resets the environment and returns a dictionary of observations 
        (keyed by the agent name)
        """
        # Randomize number of nodes and tasks within specified ranges
        num_nodes = random.randint(self.config["training_nodes_per_episode_min"], self.config["training_nodes_per_episode_max"])
        num_tasks = random.randint(self.config["training_nodes_per_episode_min"], self.config["training_nodes_per_episode_max"])

        # Update cluster
        self.config["cluster_nodes"] = num_nodes
        cluster = ClusterSynthesizer(self.config).create_cluster()
        self.agents = [f"agent_{i}" for i in range(num_nodes)]

        # Update workload
        self.config["workload_tasks"] = num_tasks
        tasks = WorkloadSynthesizer(self.config).create_tasks()

        # Update scheduler
        self.scheduler.onClusterReset(cluster)

        if self.sim_thread is not None:
            # Make sure the previous simulation thread ends cleanly
            self.coordinator.stop()
            self.sim_thread.join()
            self.coordinator.restart(main_turn_first=False)

        # Start a single simulation
        self.sim_thread = SimulatorThread(self.sim_controller, cluster, self.scheduler, tasks)
        self.sim_thread.start()

        self.coordinator.wait_for_turn(is_main=True)

        # Get the next states and build the action masks
        obs = self.scheduler.getObservations()
        infos = self._build_infos(self.scheduler.getValidNodes())
        return obs, infos

    def step(self, actions):
        """
        Receives a dictionary of actions keyed by the agent name.
        Returns the observation dictionary, reward dictionary, terminated 
        dictionary, truncated dictionary and info dictionary, where each 
        dictionary is keyed by the agent.
        """
        # Provide the actions to the scheduler and switch to the simulator
        self.scheduler.setActions(actions)
        self.coordinator.switch_turn()
        self.coordinator.wait_for_turn(is_main=True)

        # Get the next states and rewards from the scheduler
        obs = self.scheduler.getObservations()
        rewards = self.scheduler.getRewards()

        # Termination and truncation must be returned for all agents
        terminated = not self.scheduler.isSimRunning()
        terminateds = {agent: terminated for agent in self.agents}
        truncateds = {agent: False for agent in self.agents}

        # Generate valid action mask per agent
        infos = self._build_infos(self.scheduler.getValidNodes())

        if terminated and self.config.get("test_parallel_env", False):
            self.agents = []

        return obs, rewards, terminateds, truncateds, infos

    def render(self):
        """Displays a rendered frame from the environment, if supported."""
        pass

    def close(self):
        """Closes the rendering window."""
        self.coordinator.stop()
        self.sim_thread.join()

    def state(self) -> np.ndarray:
        """Returns the state.
        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX
        """
        states = self.scheduler.getObservations()
        return np.array(states.values())

    def observation_space(self, agent) -> spaces.Space:
        """Takes in agent and returns the observation space for that agent."""
        return self.observation_spaces[agent]
    
    def action_space(self, agent) -> spaces.Space:
        """Takes in agent and returns the action space for that agent."""
        return self.action_spaces[agent]
    
    @property
    def observation_spaces(self):
        return {agent: self._observation_space for agent in self.possible_agents}

    @property
    def action_spaces(self):
        return {agent: self._action_space for agent in self.possible_agents}

    def _build_infos(self, valid_nodes):
        infos = {}

        for idx, agent in enumerate(self.agents):
            if valid_nodes[idx]:
                # Full action space allowed
                action_mask = self._full_action_space_mask
            else:
                # Only the first action is allowed
                action_mask = self._zero_action_space_mask

            infos[agent] = {"action_mask": action_mask}

        return infos
    
    def getConfig(self) -> dict:
        return self.config
