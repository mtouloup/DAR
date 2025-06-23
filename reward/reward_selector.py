from reward.LB_reward import LB_reward
from reward.reward import BaseReward

class RewardSelector:
    def __init__(self, reward_config: dict, cluster):
        self.reward_config = reward_config
        self.cluster = cluster

    def create_reward(self) -> BaseReward:
        reward_type = self.reward_config.get("Type", "LB_reward")

        if reward_type == "LB_reward":
            return LB_reward(self.cluster)
        else:
            raise ValueError(f"Unsupported reward type: {reward_type}")
