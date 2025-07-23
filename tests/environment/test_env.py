import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cutsimulator.environment.env_creator import env

# Tests the Daro PZ enivornment using a number of episodes and steps
def test_env(yaml_files, num_episodes, num_steps):
    # Create the environment
    custom_env = env(yaml_files[0])
    
    for episode in range(num_episodes):
        print(f"\n*** Episode {episode + 1} ***")

        # Reset the environment and get initial observations
        observations, infos = custom_env.reset()
        print("Initial Observations:", observations)

        for step in range(num_steps):
            print(f"\n--- Step {step + 1} ---")
            
            # Random action for each agent (just for testing)
            actions = {
                agent: custom_env.action_spaces[agent].sample() if info["action_mask"][1] else 0
                for agent, info in zip(custom_env.agents, infos.values())
            }

            print("Actions:", actions)

            # Step the environment
            observations, rewards, terminateds, truncateds, infos = custom_env.step(actions)

            #print("Observations:", observations)
            print("Rewards:", rewards)
            print("Terminateds:", terminateds)

            if all(terminateds.values()):
                print("All agents are done. Ending episode.")
                break

    custom_env.close()
    print("All steps completed. Ending test.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_env.py configs.yaml")
        sys.exit(1)

    yaml_files = sys.argv[1:]
    
    test_env(yaml_files, 5, 15)
