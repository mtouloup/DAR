from cutsimulator.reward.LB_reward import LB_reward
from cutsimulator.reward.reward import BaseReward
from cutsimulator.reward.coop_lb_reward import Coop_LB_reward

class RewardSelector:
    def __init__(self, reward_config: dict, cluster):
        self.reward_config = reward_config
        self.cluster = cluster

    def create_reward(self) -> BaseReward:
        reward_type = self.reward_config.get("scheduler_daro_reward_type")

        if reward_type == "LB_reward":
            return LB_reward(self.cluster)
        elif reward_type == "Coop_LB_reward":
            return Coop_LB_reward(self.cluster)
        else:
            raise ValueError(f"Unsupported reward type: {reward_type}")
