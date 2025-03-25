import os
from vec_environment_2 import VECEnvironment

def test_env():
    print("Current working directory:", os.getcwd())
    print("Testing environment...")
    
    env_config = {
        'sumo_config': 'astana.sumocfg',
        'simulation_duration': 300,
        'time_step': 1,
        'queue_process_interval': 5,
        'max_queue_length': 50,
        'history_length': 10,
        'seed': 42
    }
    
    env = VECEnvironment(**env_config)
    obs = env.reset()
    print("Environment reset successful")
    
    # Try taking a few steps
    for i in range(5):
        print(f"Taking step {i+1}")
        action = 0  # Just use a dummy action
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1} result - reward: {reward}, done: {done}")
        if done:
            break
    
    env.close()
    print("Environment closed")

if __name__ == "__main__":
    test_env()