import cutsimulator.state.obs_builder as ob
from cutsimulator.cluster.cluster import Cluster
from cutsimulator.cluster.node import Node
from cutsimulator.reward.reward_selector import RewardSelector
from cutsimulator.scheduler.broker import Broker
from cutsimulator.scheduler.scheduler import Scheduler
from cutsimulator.workload.pod import Pod

class DaroTrainScheduler(Scheduler):

    def __init__(self, config, cluster: Cluster):

        self.num_agents = config["cluster_nodes"]
        self.epsilon = config["scheduler_daro_Epsilon"]
        self.lr = config["scheduler_daro_LearningRate"]
        self.hidden_dim = config["scheduler_daro_hidden_dims"]
        self.gamma = config["scheduler_daro_GAMMA"]
        self.update_target_every = config["scheduler_daro_Update_target_every"]
        self.double_q = config["scheduler_daro_DoubleQ"]
        self.buffer_size = config["scheduler_daro_Replay_buffer_size"]
        self.batch_size = config["scheduler_daro_BatchSize"]
        self.mixing_embed_dim = config["scheduler_daro_Mixing_embed_dim"]
        self.hypernet_layers = config["scheduler_daro_Hypernet_layers"]
        self.hypernet_embed = config["scheduler_daro_Hypernet_embed"]

        reward = RewardSelector(config, cluster).create_reward()

        self.broker = Broker(
            cluster=cluster,
            reward_fn=reward,
            num_agents=self.num_agents,
            input_dim=ob.obs_dimensions(),
            output_dim=10,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            gamma=self.gamma,
            update_target_every=self.update_target_every,
            double_q=self.double_q,
            epsilon=self.epsilon,
            mixing_embed_dim=self.mixing_embed_dim,
            hypernet_layers=self.hypernet_layers,
            hypernet_embed=self.hypernet_embed,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size
        )

    def schedule(self, pod: Pod) -> Node:
        selected_node = self.broker.schedule_pod(pod)
        return selected_node
    
    def save_model(self, path="qmix_latest.pth"):
        self.broker.save_model(path)

    def onPodDeployed(self, pod: Pod):
        self.broker.onPodDeployed(pod)

    def onPodTerminated(self, pod: Pod):
        pass

    def onSimulationEnded(self):
        print("Simulation Ended")
        self.save_model()

    def onClusterReset(self, cluster: Cluster):
        self.broker.onClusterReset(cluster)
