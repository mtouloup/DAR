from cutsimulator.environment.daro_pz_env import DaroPettingZooEnv
from cutsimulator.utils.utility import load_configs

def env(config_file: str):
    config = load_configs([config_file])
    return DaroPettingZooEnv(config)
