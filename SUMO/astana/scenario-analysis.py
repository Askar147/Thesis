#!/usr/bin/env python3
"""
Evaluation Script for FF-DQN models in VEC environment
This script focuses on evaluating just the FF-DQN model.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
import traceback
import time

from vec_environment_2 import VECEnvironment
from vec_dqn_agent_2 import DQNAgent

def flatten_observation(obs):
    """Flatten observation for DQN input"""
    flattened = []
    
    # Task information
    flattened.append(obs['task_size'][0])
    flattened.append(obs['required_cpu_cycles'][0])
    flattened.append(obs['task_deadline'][0])
    flattened.append(obs['task_scenario_id'][0])
    
    # Vehicle information
    flattened.append(obs['vehicle_pos_x'][0])
    flattened.append(obs['vehicle_pos_y'][0])
    flattened.append(obs['vehicle_speed'][0])
    
    # Base station information
    flattened.append(obs['distance_to_bs'][0])
    flattened.append(obs['bs_queue_length'][0])
    
    # Edge node information
    flattened.append(obs['active_nodes'][0])
    flattened.extend(obs['node_loads'])
    flattened.extend(obs['node_active_status'])
    
    # Historical load information
    flattened.extend(obs['historical_loads'].flatten())
    
    return np.array(flattened, dtype=np.float32)

def run_evaluation(agent, env, num_episodes=5, scenario_name="normal"):
    """Run evaluation episodes"""
    print(f"\nEvaluating scenario: {scenario_name}")
    
    rewards = []
    task_counts = []
    completion_rates = []
    rejection_rates = []
    drop_rates = []
    latencies = []
    energy_consumptions = []
    
    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")
        
        # Reset environment
        obs = env.reset()
        state = flatten_observation(obs)
        episode_reward = 0
        episode_steps = 0
        episode_task_count = 0
        episode_completions = []
        episode_rejections = []
        episode_drops = []
        episode_latencies = []
        episode_energies = []
        
        done = False
        max_steps = 300
        
        while not done and episode_steps < max_steps:
            # Select action using the agent
            action = agent.select_action(state)
            
            # Take a step in the environment
            next_obs, reward, done, info = env.step(action)
            next_state = flatten_observation(next_obs)
            
            # Track metrics
            episode_reward += reward
            episode_steps += 1
            
            # Track task-related metrics
            if info.get('task_completed', False) or info.get('task_rejected', False) or info.get('task_dropped', False):
                episode_task_count += 1
            
            if 'task_completion_rate' in info:
                episode_completions.append(info['task_completion_rate'])
            
            if 'task_rejection_rate' in info:
                episode_rejections.append(info['task_rejection_rate'])
            
            if 'task_drop_rate' in info:
                episode_drops.append(info['task_drop_rate'])
            
            if 'avg_latency' in info:
                episode_latencies.append(info['avg_latency'])
                
            if 'energy_consumption' in info:
                episode_energies.append(info['energy_consumption'])
            
            # Update state
            state = next_state
        
        # Store episode metrics
        rewards.append(episode_reward)
        task_counts.append(episode_task_count)
        
        if episode_completions:
            completion_rates.append(np.mean(episode_completions))
        
        if episode_rejections:
            rejection_rates.append(np.mean(episode_rejections))
        
        if episode_drops:
            drop_rates.append(np.mean(episode_drops))
        
        if episode_latencies:
            latencies.append(np.mean(episode_latencies))
            
        if episode_energies:
            energy_consumptions.append(np.sum(episode_energies))
            
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Tasks: {episode_task_count}")
        print(f"  Completion Rate: {np.mean(episode_completions) if episode_completions else 0:.2f}")
        print(f"  Avg Latency: {np.mean(episode_latencies) if episode_latencies else 0:.4f}s")
        print(f"  Energy: {np.sum(episode_energies) if episode_energies else 0:.2f}J")
    
    # Calculate aggregate metrics
    metrics = {
        'scenario': scenario_name,
        'rewards': rewards,
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'task_counts': task_counts,
        'avg_task_count': np.mean(task_counts) if task_counts else 0,
        'completion_rates': completion_rates,
        'avg_completion_rate': np.mean(completion_rates) if completion_rates else 0,
        'std_completion_rate': np.std(completion_rates) if completion_rates else 0,
        'rejection_rates': rejection_rates,
        'avg_rejection_rate': np.mean(rejection_rates) if rejection_rates else 0,
        'drop_rates': drop_rates,
        'avg_drop_rate': np.mean(drop_rates) if drop_rates else 0,
        'latencies': latencies,
        'avg_latency': np.mean(latencies) if latencies else 0,
        'std_latency': np.std(latencies) if latencies else 0,
        'energy_consumptions': energy_consumptions,
        'avg_energy': np.mean(energy_consumptions) if energy_consumptions else 0,
        'std_energy': np.std(energy_consumptions) if energy_consumptions else 0
    }
    
    return metrics

def analyze_model(model_path, output_dir, energy_csv_path=None, num_episodes=3):
    """Analyze FF-DQN model's performance across different scenarios"""
    print(f"\nAnalyzing FF-DQN model from {model_path}")
    
    # Create result directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"ff_dqn_eval_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Define base environment config
    base_env_config = {
        'sumo_config': 'astana.sumocfg',
        'simulation_duration': 300,
        'time_step': 1,
        'queue_process_interval': 5,
        'max_queue_length': 50,
        'history_length': 10,
        'energy_csv_path': energy_csv_path,
        'energy_weight': 0.5,
        'seed': 42
    }
    
    # Define scenarios with modified configs
    scenarios = {
        'normal': base_env_config.copy(),
        'high_load': dict(base_env_config, **{'simulation_duration': 500}),  # Longer simulation
        'low_latency': dict(base_env_config, **{'energy_weight': 0.3}),      # Prioritize latency
        'energy_efficient': dict(base_env_config, **{'energy_weight': 0.7})  # Prioritize energy
    }
    
    # Create initial environment to get state/action sizes
    env = VECEnvironment(**base_env_config)
    obs = env.reset()
    state = flatten_observation(obs)
    state_size = len(state)
    action_size = env.action_space.n
    print(f"State size: {state_size}, Action size: {action_size}")
    env.close()
    
    # Initialize agent
    agent = DQNAgent(state_size, action_size)
    
    # Load model
    success = agent.load_model(model_path)
    if not success:
        print(f"Failed to load model from {model_path}")
        return None
    
    # Disable exploration
    agent.epsilon = 0.0
    
    # Run evaluations for each scenario
    results = {}
    
    for scenario_name, config in scenarios.items():
        # Create environment with this config
        env = VECEnvironment(**config)
        
        # Run evaluation
        results[scenario_name] = run_evaluation(agent, env, num_episodes, scenario_name)
        
        # Close environment
        env.close()
    
    # Save results
    results_file = os.path.join(results_dir, "analysis_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Generate plots
    plot_model_results(results, results_dir)
    
    print(f"Analysis completed for FF-DQN")
    print(f"Results saved to {results_dir}")
    
    return results

def plot_model_results(results, save_dir):
    """Generate plots for model results"""
    # Get scenario names
    scenarios = list(results.keys())
    
    # Plot key metrics
    metrics = [
        ('avg_reward', 'Average Reward'),
        ('avg_completion_rate', 'Task Completion Rate', lambda x: x * 100),
        ('avg_latency', 'Average Latency (s)'),
        ('avg_energy', 'Energy Consumption (J)')
    ]
    
    plt.figure(figsize=(15, 10))
    
    for i, metric_info in enumerate(metrics):
        metric, title = metric_info[:2]
        transform_func = metric_info[2] if len(metric_info) > 2 else None
        
        plt.subplot(2, 2, i+1)
        
        # Extract values
        values = []
        errors = []
        
        for scenario in scenarios:
            val = results[scenario].get(metric, 0)
            if transform_func:
                val = transform_func(val)
            values.append(val)
            
            # Get error if available
            std_key = f"std_{metric.replace('avg_', '')}"
            if std_key in results[scenario]:
                err = results[scenario][std_key]
                if transform_func:
                    err = transform_func(err)
                errors.append(err)
            else:
                errors.append(0)
        
        # Create bar chart
        plt.bar(scenarios, values, yerr=errors, alpha=0.7)
        plt.title(title)
        plt.ylabel(title)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle(f"FF-DQN Performance Across Scenarios", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, "performance_summary.png"), dpi=300)
    plt.close()
    
    # Create CSV summary
    summary_data = []
    
    for scenario in scenarios:
        data = results[scenario]
        
        row = {
            'Scenario': scenario,
            'Reward': data.get('avg_reward', 0),
            'Task Count': data.get('avg_task_count', 0),
            'Completion Rate (%)': data.get('avg_completion_rate', 0) * 100,
            'Rejection Rate (%)': data.get('avg_rejection_rate', 0) * 100,
            'Drop Rate (%)': data.get('avg_drop_rate', 0) * 100,
            'Latency (s)': data.get('avg_latency', 0),
            'Energy (J)': data.get('avg_energy', 0)
        }
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(save_dir, "results_summary.csv"), index=False)
    
    # Print summary
    print("\nPerformance Summary:")
    for scenario in scenarios:
        data = results[scenario]
        print(f"  {scenario}:")
        print(f"    Reward: {data.get('avg_reward', 0):.2f}")
        print(f"    Completion Rate: {data.get('avg_completion_rate', 0)*100:.2f}%")
        print(f"    Latency: {data.get('avg_latency', 0):.4f}s")
        print(f"    Energy: {data.get('avg_energy', 0):.2f}J")
    
    print("\nResults saved to:", save_dir)

def main():
    parser = argparse.ArgumentParser(description='Evaluate FF-DQN model in VEC environment')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to the trained FF-DQN model checkpoint')
    parser.add_argument('--output_dir', type=str, default='dqn_evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--energy_csv_path', type=str, default=None,
                       help='Path to energy consumption data CSV (optional)')
    parser.add_argument('--num_episodes', type=int, default=3,
                       help='Number of episodes to run for each scenario')
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    try:
        start_time = time.time()
        
        # Analyze FF-DQN model
        results = analyze_model(
            args.model, 
            args.output_dir,
            args.energy_csv_path,
            args.num_episodes
        )
        
        # Print timing information
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\nEvaluation completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()