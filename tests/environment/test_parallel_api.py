import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pettingzoo.test import parallel_api_test

from cutsimulator.environment.env_creator import env

# Tests the Daro PZ enivornment using an API test provided by PettingZoo
def test_parallel_env(yaml_files):
    # Create the environment
    custom_env = env(yaml_files[0])

    # Make sure enough pods are generated for the test
    num_cycles=100
    config = custom_env.getConfig()
    config["training_tasks_per_episode_min"] = num_cycles + 1
    config["training_tasks_per_episode_max"] = num_cycles + 1
    config["test_parallel_env"] = True

    # Perform the api test
    parallel_api_test(custom_env, num_cycles)
    custom_env.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_parallel_api.py configs.yaml")
        sys.exit(1)

    yaml_files = sys.argv[1:]
    
    test_parallel_env(yaml_files)
