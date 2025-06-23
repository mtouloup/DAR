from cluster.cluster import Cluster
from cluster.node import Node
from reward.reward_selector import RewardSelector
from scheduler.broker import Broker
from scheduler.scheduler import Scheduler
from workload.pod import Pod

class DAROScheduler(Scheduler):

    def __init__(self, config, cluster: Cluster):

        self.num_agents = config["Cluster"]["Nodes"]["Number"]
        self.epsilon = config["Scheduler"]["Params"]["Epsilon"]
        self.lr = config["Scheduler"]["Params"]["LearningRate"]
        self.hidden_dim = config["Scheduler"]["Params"]["hidden_dims"]
        self.gamma = config["Scheduler"]["Params"]["GAMMA"]
        self.update_target_every = config["Scheduler"]["Params"]["Update_target_every"]
        self.double_q = config["Scheduler"]["Params"]["DoubleQ"]
        self.buffer_size = config["Scheduler"]["Params"]["Replay_buffer_size"]
        self.batch_size = config["Scheduler"]["Params"]["BatchSize"]
        self.mixing_embed_dim = config["Scheduler"]["Params"]["Mixing_embed_dim"]
        self.hypernet_layers = config["Scheduler"]["Params"]["Hypernet_layers"]
        self.hypernet_embed = config["Scheduler"]["Params"]["Hypernet_embed"]
        reward_config = config["Scheduler"].get("Reward", {"Type": "LB_reward"})
        reward = RewardSelector(reward_config, cluster).create_reward()

        self.broker = Broker(
            cluster=cluster,
            reward_fn=reward,
            num_agents=self.num_agents,
            input_dim=8,
            output_dim=11,
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
