import numpy as np
import matplotlib.pyplot as plt
from vec_environment_2 import VECEnvironment
from energy_model import EnergyModel

def test_energy_model():
    """Test the energy consumption model"""
    print("Testing Energy Model...")
    
    # Create energy model
    energy_model = EnergyModel("merged_dag1.csv")
    
    # Print average power for each scenario
    print("\nAverage Power Consumption (Watts):")
    for scenario_id in range(1, 11):
        power = energy_model.get_average_power(scenario_id)
        print(f"Scenario {scenario_id}: {power:.2f} W")
    
    # Calculate energy consumption for different durations
    print("\nEnergy Consumption (Joules) for 1s task:")
    for scenario_id in range(1, 11):
        energy = energy_model.get_energy_consumption(scenario_id, duration=1.0)
        print(f"Scenario {scenario_id}: {energy:.2f} J")

def test_vec_environment():
    """Test the VEC environment with energy consumption"""
    print("\nTesting VEC Environment with Energy Consumption...")
    
    # Create environment
    env = VECEnvironment(
        sumo_config="astana.sumocfg",
        simulation_duration=300,
        energy_csv_path="merged_dag1.csv",
        energy_weight=0.5
    )
    
    # Reset environment
    obs = env.reset()
    print(f"Initial observation keys: {obs.keys()}")
    
    # Test environment for 100 steps
    total_reward = 0
    latencies = []
    energy_values = []
    completion_rates = []
    
    for step in range(100):
        # Take random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        total_reward += reward
        
        # Track metrics
        if 'avg_latency' in info and info['avg_latency'] > 0:
            latencies.append(info['avg_latency'])
        
        if 'avg_energy_consumption' in info and info['avg_energy_consumption'] > 0:
            energy_values.append(info['avg_energy_consumption'])
        
        if 'task_completion_rate' in info:
            completion_rates.append(info['task_completion_rate'])
        
        # Print some info every 10 steps
        if step % 10 == 0:
            print(f"Step {step}: Reward: {reward:.4f}, " +
                  f"Avg Latency: {info.get('avg_latency', 0):.4f}, " + 
                  f"Avg Energy: {info.get('avg_energy_consumption', 0):.4f}, " +
                  f"Completion Rate: {info.get('task_completion_rate', 0):.4f}")
        
        if done:
            print(f"Environment done after {step+1} steps")
            break
    
    # Close environment
    env.close()
    
    print(f"\nTotal reward: {total_reward:.4f}")
    
    # Plot results if we have data
    if latencies and energy_values and completion_rates:
        plt.figure(figsize=(15, 5))
        
        # Plot latencies
        plt.subplot(1, 3, 1)
        plt.plot(latencies)
        plt.title('Average Latency')
        plt.xlabel('Step')
        plt.ylabel('Latency (s)')
        
        # Plot energy consumption
        plt.subplot(1, 3, 2)
        plt.plot(energy_values)
        plt.title('Average Energy Consumption')
        plt.xlabel('Step')
        plt.ylabel('Energy (J)')
        
        # Plot completion rate
        plt.subplot(1, 3, 3)
        plt.plot(completion_rates)
        plt.title('Task Completion Rate')
        plt.xlabel('Step')
        plt.ylabel('Rate')
        
        plt.tight_layout()
        plt.savefig('vec_environment_test.png')
        plt.show()

def run_different_energy_weights():
    """Test the VEC environment with different energy weights"""
    print("\nTesting VEC Environment with Different Energy Weights...")
    
    # Energy weights to test
    energy_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    num_steps = 200
    
    results = {}
    
    for weight in energy_weights:
        print(f"\nTesting energy weight: {weight}")
        
        # Create environment with this weight
        env = VECEnvironment(
            sumo_config="astana.sumocfg",
            simulation_duration=300,
            energy_csv_path="merged_dag1.csv",
            energy_weight=weight
        )
        
        # Reset environment
        obs = env.reset()
        
        # Run environment for num_steps
        total_reward = 0
        avg_latency = 0
        avg_energy = 0
        completion_rate = 0
        step_count = 0
        
        for step in range(num_steps):
            # Take random action
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Track last metrics
            if 'avg_latency' in info:
                avg_latency = info['avg_latency']
            
            if 'avg_energy_consumption' in info:
                avg_energy = info['avg_energy_consumption']
            
            if 'task_completion_rate' in info:
                completion_rate = info['task_completion_rate']
            
            if done:
                print(f"Environment done after {step+1} steps")
                break
        
        # Close environment
        env.close()
        
        # Store results
        results[weight] = {
            'reward': total_reward / step_count,
            'latency': avg_latency,
            'energy': avg_energy,
            'completion_rate': completion_rate
        }
        
        print(f"Weight {weight}: Avg Reward: {total_reward / step_count:.4f}, " +
              f"Avg Latency: {avg_latency:.4f}, " + 
              f"Avg Energy: {avg_energy:.4f}, " +
              f"Completion Rate: {completion_rate:.4f}")
    
    # Plot results
    weights = list(results.keys())
    rewards = [results[w]['reward'] for w in weights]
    latencies = [results[w]['latency'] for w in weights]
    energies = [results[w]['energy'] for w in weights]
    completion_rates = [results[w]['completion_rate'] for w in weights]
    
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(weights, rewards, 'o-')
    plt.title('Average Reward vs Energy Weight')
    plt.xlabel('Energy Weight')
    plt.ylabel('Average Reward')
    
    # Plot latencies
    plt.subplot(2, 2, 2)
    plt.plot(weights, latencies, 'o-')
    plt.title('Average Latency vs Energy Weight')
    plt.xlabel('Energy Weight')
    plt.ylabel('Latency (s)')
    
    # Plot energy consumption
    plt.subplot(2, 2, 3)
    plt.plot(weights, energies, 'o-')
    plt.title('Average Energy Consumption vs Energy Weight')
    plt.xlabel('Energy Weight')
    plt.ylabel('Energy (J)')
    
    # Plot completion rates
    plt.subplot(2, 2, 4)
    plt.plot(weights, completion_rates, 'o-')
    plt.title('Task Completion Rate vs Energy Weight')
    plt.xlabel('Energy Weight')
    plt.ylabel('Completion Rate')
    
    plt.tight_layout()
    plt.savefig('energy_weight_comparison.png')
    plt.show()

if __name__ == "__main__":
    # Test energy model
    test_energy_model()
    
    # Test VEC environment
    test_vec_environment()
    
    # Test different energy weights
    # This is more extensive and optional
    # run_different_energy_weights()