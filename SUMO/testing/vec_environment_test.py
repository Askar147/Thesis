#!/usr/bin/env python3
"""
Test script to verify the integration of enhanced energy model and simplified latency model
in the VEC environment
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from vec_environment import VECEnvironment

def test_environment():
    """Test the enhanced VEC environment with integrated models"""
    
    print("Creating environment with enhanced energy and latency models...")
    env = VECEnvironment(
        sumo_config="astana.sumocfg",
        simulation_duration=300,  # 5 minutes of simulation
        time_step=1,
        queue_process_interval=5,
        energy_csv_path="merged_dag1.csv",
        energy_weight=0.5,
        latency_model_params={
            'frequency_band': 2.4,
            'bandwidth': 20,
            'noise_floor': -95
        },
        seed=42
    )
    
    # Track metrics
    rewards = []
    energy_consumption = []
    latencies = []
    task_completion_rates = []
    data_rates = []
    
    # Reset the environment
    print("Resetting environment...")
    obs = env.reset()
    done = False
    total_steps = 0
    
    # Take steps with random actions
    print("Taking random actions...")
    while not done and total_steps < 100:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        total_steps += 1
        
        # Track metrics
        rewards.append(reward)
        energy_consumption.append(info.get('energy_consumption', 0))
        latencies.append(info.get('avg_latency', 0))
        task_completion_rates.append(info.get('task_completion_rate', 0))
        data_rates.append(info.get('avg_data_rate', 0))
        
        # Print info every 10 steps
        if total_steps % 10 == 0:
            print(f"Step {total_steps}:")
            print(f"  Reward: {reward:.4f}")
            print(f"  Task Completion Rate: {info.get('task_completion_rate', 0):.2f}")
            print(f"  Avg Latency: {info.get('avg_latency', 0):.4f}s")
            print(f"  Energy Consumption: {info.get('energy_consumption', 0):.2f}J")
            print(f"  Avg Data Rate: {info.get('avg_data_rate', 0):.2f} Mbps")
            print(f"  Idle Energy: {info.get('idle_energy', 0):.2f}J")
    
    env.close()
    print(f"Environment closed after {total_steps} steps")
    
    # Plot results
    plot_results(rewards, energy_consumption, latencies, task_completion_rates, data_rates)
    
    return total_steps

def plot_results(rewards, energy_consumption, latencies, task_completion_rates, data_rates):
    """Plot the results of the test"""
    fig, axs = plt.subplots(5, 1, figsize=(10, 15))
    
    # Plot rewards
    axs[0].plot(rewards)
    axs[0].set_title('Rewards')
    axs[0].set_xlabel('Step')
    axs[0].set_ylabel('Reward')
    
    # Plot energy consumption
    axs[1].plot(energy_consumption)
    axs[1].set_title('Energy Consumption')
    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('Energy (J)')
    
    # Plot latencies
    axs[2].plot(latencies)
    axs[2].set_title('Average Latency')
    axs[2].set_xlabel('Step')
    axs[2].set_ylabel('Latency (s)')
    
    # Plot task completion rate
    axs[3].plot(task_completion_rates)
    axs[3].set_title('Task Completion Rate')
    axs[3].set_xlabel('Step')
    axs[3].set_ylabel('Rate')
    
    # Plot data rates
    axs[4].plot(data_rates)
    axs[4].set_title('Average Data Rate')
    axs[4].set_xlabel('Step')
    axs[4].set_ylabel('Data Rate (Mbps)')
    
    plt.tight_layout()
    plt.savefig('vec_environment_test_results.png')
    plt.close()
    
    print("Results plotted and saved to 'vec_environment_test_results.png'")

if __name__ == "__main__":
    test_environment()