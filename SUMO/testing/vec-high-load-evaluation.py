#!/usr/bin/env python3
"""
Evaluation script for VEC task offloading models under high-load conditions.
Compares TE-DQN and FF-DQN performance with significantly increased task loads.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# Import environment and agents
from vec_environment import VECEnvironment
from te_dqn_agent import VECTransformerAgent, flatten_observation as transformer_flatten
from ff_dqn_agent import DQNAgent


def get_state_size(env, model_type):
    """Calculate state size for a specific model type"""
    obs = env.reset()
    if model_type == "transformer":
        return len(transformer_flatten(obs))
    else:  # dqn
        agent = DQNAgent(0, env.action_space.n)
        return len(agent.flatten_observation(obs))


def create_agent(model_type, state_size, action_size):
    """Create appropriate agent based on model type"""
    if model_type == "transformer":
        return VECTransformerAgent(state_size, action_size)
    else:  # dqn
        return DQNAgent(state_size, action_size)


def load_model(agent, model_path):
    """Load model weights from path"""
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return False
    
    try:
        agent.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False


def evaluate_model(env, agent, num_episodes, model_type, max_steps=1000, verbose=True):
    """
    Evaluate a single model's performance
    
    Args:
        env: VEC environment instance
        agent: Loaded agent (TE-DQN or FF-DQN)
        num_episodes: Number of evaluation episodes
        model_type: "transformer" or "dqn"
        max_steps: Maximum steps per episode
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Metrics to track
    episode_rewards = []
    episode_steps = []
    task_completion_rates = []
    task_rejection_rates = []
    task_drop_rates = []
    avg_latencies = []
    energy_consumptions = []
    idle_energies = []
    avg_data_rates = []
    node_utilizations = []
    
    # Create progress bar
    progress = tqdm(range(num_episodes), desc=f"Evaluating {model_type.upper()}")
    
    for episode in progress:
        # Reset environment and agent state
        obs = env.reset()
        if model_type == "transformer":
            state = transformer_flatten(obs)
            agent.state_history.clear()  # Clear state history for transformer
        else:
            state = agent.flatten_observation(obs)
        
        # Track episode metrics
        episode_reward = 0
        steps = 0
        completion_rate_history = []
        rejection_rate_history = []
        drop_rate_history = []
        latency_history = []
        energy_history = []
        idle_energy_history = []
        data_rate_history = []
        node_usage_history = []
        
        # Run episode
        done = False
        while not done and steps < max_steps:
            # Select action deterministically
            if model_type == "transformer":
                action = agent.select_action(state, evaluate=True)
            else:
                # Temporarily set epsilon to 0 for deterministic action selection
                old_epsilon = agent.epsilon
                agent.epsilon = 0
                action = agent.select_action(state)
                agent.epsilon = old_epsilon
            
            # Execute action
            next_obs, reward, done, info = env.step(action)
            
            # Process next state
            if model_type == "transformer":
                next_state = transformer_flatten(next_obs)
            else:
                next_state = agent.flatten_observation(next_obs)
            
            # Update metrics
            episode_reward += reward
            steps += 1
            
            # Track detailed metrics from info
            if 'task_completion_rate' in info:
                completion_rate_history.append(info['task_completion_rate'])
            if 'task_rejection_rate' in info:
                rejection_rate_history.append(info['task_rejection_rate'])
            if 'task_drop_rate' in info:
                drop_rate_history.append(info['task_drop_rate'])
            if 'avg_latency' in info:
                latency_history.append(info['avg_latency'])
            if 'avg_energy_consumption' in info:
                energy_history.append(info['avg_energy_consumption'])
            if 'idle_energy' in info:
                idle_energy_history.append(info['idle_energy'])
            if 'avg_data_rate' in info:
                data_rate_history.append(info['avg_data_rate'])
            
            # Calculate node utilization
            node_usage = 0
            node_counts = 0
            for bs_id, bs_instance in env.base_station_instances.items():
                active_count = sum(1 for node in bs_instance.nodes if node.active)
                node_usage += active_count / len(bs_instance.nodes)
                node_counts += 1
            if node_counts > 0:
                node_usage_history.append(node_usage / node_counts)
            
            state = next_state
        
        # Collect episode metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        
        # Calculate average metrics for this episode
        if completion_rate_history:
            task_completion_rates.append(np.mean(completion_rate_history))
        if rejection_rate_history:
            task_rejection_rates.append(np.mean(rejection_rate_history))
        if drop_rate_history:
            task_drop_rates.append(np.mean(drop_rate_history))
        if latency_history:
            avg_latencies.append(np.mean(latency_history))
        if energy_history:
            energy_consumptions.append(np.mean(energy_history))
        if idle_energy_history:
            idle_energies.append(np.mean(idle_energy_history))
        if data_rate_history:
            avg_data_rates.append(np.mean(data_rate_history))
        if node_usage_history:
            node_utilizations.append(np.mean(node_usage_history))
        
        # Update progress bar
        progress.set_postfix({
            'reward': f"{episode_reward:.2f}", 
            'completion': f"{task_completion_rates[-1]:.2f}" if task_completion_rates else "N/A"
        })
    
    # Calculate final metrics
    results = {
        'num_episodes': num_episodes,
        'avg_reward': float(np.mean(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'avg_steps': float(np.mean(episode_steps)),
        'avg_completion_rate': float(np.mean(task_completion_rates)) if task_completion_rates else 0,
        'avg_rejection_rate': float(np.mean(task_rejection_rates)) if task_rejection_rates else 0,
        'avg_drop_rate': float(np.mean(task_drop_rates)) if task_drop_rates else 0,
        'avg_latency': float(np.mean(avg_latencies)) if avg_latencies else 0,
        'avg_energy_consumption': float(np.mean(energy_consumptions)) if energy_consumptions else 0,
        'avg_idle_energy': float(np.mean(idle_energies)) if idle_energies else 0,
        'avg_data_rate': float(np.mean(avg_data_rates)) if avg_data_rates else 0,
        'avg_node_utilization': float(np.mean(node_utilizations)) if node_utilizations else 0,
        'all_rewards': episode_rewards,
        'all_completion_rates': task_completion_rates,
        'all_rejection_rates': task_rejection_rates,
        'all_latencies': avg_latencies,
        'all_energy_consumptions': energy_consumptions,
        'all_node_utilizations': node_utilizations
    }
    
    # Print summary if verbose
    if verbose:
        print(f"\n{model_type.upper()} Model Evaluation Summary:")
        print(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Completion Rate: {results['avg_completion_rate']:.4f}")
        print(f"Rejection Rate: {results['avg_rejection_rate']:.4f}")
        print(f"Latency: {results['avg_latency']:.4f} seconds")
        print(f"Energy Consumption: {results['avg_energy_consumption']:.4f} joules")
        print(f"Node Utilization: {results['avg_node_utilization']:.4f}")
    
    return results


def compare_models(transformer_results, dqn_results, save_dir):
    """Generate comparison plots and save results"""
    # Create comparison table
    comparison = {
        'Metric': [
            'Average Reward',
            'Task Completion Rate',
            'Task Rejection Rate',
            'Average Latency (s)',
            'Energy Consumption (J)',
            'Node Utilization',
            'Average Steps/Episode'
        ],
        'TE-DQN': [
            f"{transformer_results['avg_reward']:.2f} ± {transformer_results['std_reward']:.2f}",
            f"{transformer_results['avg_completion_rate']:.4f}",
            f"{transformer_results['avg_rejection_rate']:.4f}",
            f"{transformer_results['avg_latency']:.4f}",
            f"{transformer_results['avg_energy_consumption']:.4f}",
            f"{transformer_results['avg_node_utilization']:.4f}",
            f"{transformer_results['avg_steps']:.2f}"
        ],
        'FF-DQN': [
            f"{dqn_results['avg_reward']:.2f} ± {dqn_results['std_reward']:.2f}",
            f"{dqn_results['avg_completion_rate']:.4f}",
            f"{dqn_results['avg_rejection_rate']:.4f}",
            f"{dqn_results['avg_latency']:.4f}",
            f"{dqn_results['avg_energy_consumption']:.4f}",
            f"{dqn_results['avg_node_utilization']:.4f}",
            f"{dqn_results['avg_steps']:.2f}"
        ],
        'Difference (%)': [
            f"{(transformer_results['avg_reward'] - dqn_results['avg_reward']) / max(abs(dqn_results['avg_reward']), 1e-10) * 100:.2f}%",
            f"{(transformer_results['avg_completion_rate'] - dqn_results['avg_completion_rate']) / max(abs(dqn_results['avg_completion_rate']), 1e-10) * 100:.2f}%",
            f"{(transformer_results['avg_rejection_rate'] - dqn_results['avg_rejection_rate']) / max(abs(dqn_results['avg_rejection_rate']), 1e-10) * 100:.2f}%",
            f"{(transformer_results['avg_latency'] - dqn_results['avg_latency']) / max(abs(dqn_results['avg_latency']), 1e-10) * 100:.2f}%",
            f"{(transformer_results['avg_energy_consumption'] - dqn_results['avg_energy_consumption']) / max(abs(dqn_results['avg_energy_consumption']), 1e-10) * 100:.2f}%",
            f"{(transformer_results['avg_node_utilization'] - dqn_results['avg_node_utilization']) / max(abs(dqn_results['avg_node_utilization']), 1e-10) * 100:.2f}%",
            f"{(transformer_results['avg_steps'] - dqn_results['avg_steps']) / max(abs(dqn_results['avg_steps']), 1e-10) * 100:.2f}%"
        ]
    }
    
    # Print comparison table
    df = pd.DataFrame(comparison)
    print("\nModel Comparison under High Load:")
    print(df.to_string(index=False))
    
    # Save comparison as CSV
    df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
    
    # Create comparison plots
    plt.figure(figsize=(20, 15))
    
    # 1. Rewards Distribution
    plt.subplot(2, 3, 1)
    plt.boxplot([transformer_results['all_rewards'], dqn_results['all_rewards']], 
               labels=['TE-DQN', 'FF-DQN'])
    plt.title('Reward Distribution')
    plt.grid(True, alpha=0.3)
    
    # 2. Completion Rate Comparison
    plt.subplot(2, 3, 2)
    plt.boxplot([transformer_results['all_completion_rates'], dqn_results['all_completion_rates']], 
               labels=['TE-DQN', 'FF-DQN'])
    plt.title('Task Completion Rate')
    plt.grid(True, alpha=0.3)
    
    # 3. Rejection Rate Comparison
    plt.subplot(2, 3, 3)
    plt.boxplot([transformer_results['all_rejection_rates'], dqn_results['all_rejection_rates']], 
               labels=['TE-DQN', 'FF-DQN'])
    plt.title('Task Rejection Rate')
    plt.grid(True, alpha=0.3)
    
    # 4. Latency Comparison
    plt.subplot(2, 3, 4)
    plt.boxplot([transformer_results['all_latencies'], dqn_results['all_latencies']], 
               labels=['TE-DQN', 'FF-DQN'])
    plt.title('Average Latency (s)')
    plt.grid(True, alpha=0.3)
    
    # 5. Energy Consumption Comparison
    plt.subplot(2, 3, 5)
    plt.boxplot([transformer_results['all_energy_consumptions'], dqn_results['all_energy_consumptions']], 
               labels=['TE-DQN', 'FF-DQN'])
    plt.title('Energy Consumption (J)')
    plt.grid(True, alpha=0.3)
    
# 6. Node Utilization Comparison
    plt.subplot(2, 3, 6)
    plt.boxplot([transformer_results['all_node_utilizations'], dqn_results['all_node_utilizations']], 
               labels=['TE-DQN', 'FF-DQN'])
    plt.title('Node Utilization')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'high_load_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Create a second figure for episode-by-episode trends if we have enough episodes
    if len(transformer_results['all_rewards']) > 5:
        plt.figure(figsize=(15, 10))
        
        # Plot rewards across episodes
        plt.subplot(2, 2, 1)
        plt.plot(transformer_results['all_rewards'], 'b-', label='TE-DQN', alpha=0.7)
        plt.plot(dqn_results['all_rewards'], 'r-', label='FF-DQN', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Rewards per Episode')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot completion rates across episodes
        plt.subplot(2, 2, 2)
        plt.plot(transformer_results['all_completion_rates'], 'b-', label='TE-DQN', alpha=0.7)
        plt.plot(dqn_results['all_completion_rates'], 'r-', label='FF-DQN', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Completion Rate')
        plt.title('Task Completion Rate per Episode')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot latencies across episodes
        plt.subplot(2, 2, 3)
        plt.plot(transformer_results['all_latencies'], 'b-', label='TE-DQN', alpha=0.7)
        plt.plot(dqn_results['all_latencies'], 'r-', label='FF-DQN', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Latency (s)')
        plt.title('Average Latency per Episode')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot energy consumption across episodes
        plt.subplot(2, 2, 4)
        plt.plot(transformer_results['all_energy_consumptions'], 'b-', label='TE-DQN', alpha=0.7)
        plt.plot(dqn_results['all_energy_consumptions'], 'r-', label='FF-DQN', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Energy (J)')
        plt.title('Energy Consumption per Episode')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'episode_trends.png'), dpi=300, bbox_inches='tight')
    
    plt.close('all')
    print(f"Comparison plots saved to {save_dir}")


def create_high_load_env(env_config, load_multiplier):
    """Create a high-load environment by modifying task generation parameters"""
    # Make a copy of the original config to avoid modifying it
    high_load_config = env_config.copy()
    
    # Increase task generation frequency and count
    high_load_config['min_tasks_per_step'] = max(1, int(env_config.get('min_tasks_per_step', 1) * load_multiplier))
    high_load_config['max_tasks_per_step'] = max(2, int(env_config.get('max_tasks_per_step', 5) * load_multiplier))
    
    # Ensure high task generation probability
    high_load_config['task_generation_probability'] = min(1.0, env_config.get('task_generation_probability', 0.8) * 1.25)
    
    # Create the environment
    env = VECEnvironment(**high_load_config)
    
    print(f"Created high-load environment with:")
    print(f"- Min tasks per step: {high_load_config['min_tasks_per_step']}")
    print(f"- Max tasks per step: {high_load_config['max_tasks_per_step']}")
    print(f"- Task generation probability: {high_load_config['task_generation_probability']:.2f}")
    
    return env

def run_varying_load_evaluation(transformer_model_path, dqn_model_path, env_config, 
                               load_factors=[1.0, 1.5, 2.0, 3.0, 4.0],
                               num_episodes=5, save_dir="high_load_results"):
    """Run evaluation with progressively higher loads"""
    # Create the output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Results to track for each load factor
    results = {
        'load_factors': load_factors,
        'transformer': {
            'avg_rewards': [],
            'completion_rates': [],
            'rejection_rates': [],
            'latencies': [],
            'energy_consumptions': [],
            'node_utilizations': []
        },
        'dqn': {
            'avg_rewards': [],
            'completion_rates': [],
            'rejection_rates': [],
            'latencies': [],
            'energy_consumptions': [],
            'node_utilizations': []
        }
    }
    
    # For each load factor
    for load_factor in load_factors:
        print(f"\n{'='*50}")
        print(f"Evaluating with load factor: {load_factor}x")
        print(f"{'='*50}")
        
        # Create environment with the current load factor
        evaluation_env = create_high_load_env(env_config, load_factor)
        
        # Create and evaluate the transformer agent
        transformer_state_size = get_state_size(evaluation_env, "transformer")
        transformer_agent = create_agent("transformer", transformer_state_size, evaluation_env.action_space.n)
        if load_model(transformer_agent, transformer_model_path):
            transformer_results = evaluate_model(
                evaluation_env, 
                transformer_agent, 
                num_episodes=num_episodes,
                model_type="transformer"
            )
            
            # Store results
            results['transformer']['avg_rewards'].append(transformer_results['avg_reward'])
            results['transformer']['completion_rates'].append(transformer_results['avg_completion_rate'])
            results['transformer']['rejection_rates'].append(transformer_results['avg_rejection_rate'])
            results['transformer']['latencies'].append(transformer_results['avg_latency'])
            results['transformer']['energy_consumptions'].append(transformer_results['avg_energy_consumption'])
            results['transformer']['node_utilizations'].append(transformer_results['avg_node_utilization'])
        
        # Close SUMO to release the connection
        evaluation_env.close()
        
        # Create and evaluate the DQN agent with a fresh environment
        evaluation_env = create_high_load_env(env_config, load_factor)
        dqn_state_size = get_state_size(evaluation_env, "dqn")
        dqn_agent = create_agent("dqn", dqn_state_size, evaluation_env.action_space.n)
        if load_model(dqn_agent, dqn_model_path):
            dqn_results = evaluate_model(
                evaluation_env, 
                dqn_agent, 
                num_episodes=num_episodes, 
                model_type="dqn"
            )
            
            # Store results
            results['dqn']['avg_rewards'].append(dqn_results['avg_reward'])
            results['dqn']['completion_rates'].append(dqn_results['avg_completion_rate'])
            results['dqn']['rejection_rates'].append(dqn_results['avg_rejection_rate'])
            results['dqn']['latencies'].append(dqn_results['avg_latency'])
            results['dqn']['energy_consumptions'].append(dqn_results['avg_energy_consumption'])
            results['dqn']['node_utilizations'].append(dqn_results['avg_node_utilization'])
        
        # Close SUMO again
        evaluation_env.close()
    
    # Create plots for varying load factors
    plt.figure(figsize=(20, 15))
    
    # 1. Average Rewards vs Load
    plt.subplot(2, 3, 1)
    plt.plot(load_factors, results['transformer']['avg_rewards'], 'bo-', label='TE-DQN')
    plt.plot(load_factors, results['dqn']['avg_rewards'], 'ro-', label='FF-DQN')
    plt.xlabel('Load Factor')
    plt.ylabel('Average Reward')
    plt.title('Reward vs. Load Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Completion Rate vs Load
    plt.subplot(2, 3, 2)
    plt.plot(load_factors, results['transformer']['completion_rates'], 'bo-', label='TE-DQN')
    plt.plot(load_factors, results['dqn']['completion_rates'], 'ro-', label='FF-DQN')
    plt.xlabel('Load Factor')
    plt.ylabel('Completion Rate')
    plt.title('Task Completion Rate vs. Load Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Rejection Rate vs Load
    plt.subplot(2, 3, 3)
    plt.plot(load_factors, results['transformer']['rejection_rates'], 'bo-', label='TE-DQN')
    plt.plot(load_factors, results['dqn']['rejection_rates'], 'ro-', label='FF-DQN')
    plt.xlabel('Load Factor')
    plt.ylabel('Rejection Rate')
    plt.title('Task Rejection Rate vs. Load Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Latency vs Load
    plt.subplot(2, 3, 4)
    plt.plot(load_factors, results['transformer']['latencies'], 'bo-', label='TE-DQN')
    plt.plot(load_factors, results['dqn']['latencies'], 'ro-', label='FF-DQN')
    plt.xlabel('Load Factor')
    plt.ylabel('Latency (s)')
    plt.title('Average Latency vs. Load Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Energy Consumption vs Load
    plt.subplot(2, 3, 5)
    plt.plot(load_factors, results['transformer']['energy_consumptions'], 'bo-', label='TE-DQN')
    plt.plot(load_factors, results['dqn']['energy_consumptions'], 'ro-', label='FF-DQN')
    plt.xlabel('Load Factor')
    plt.ylabel('Energy (J)')
    plt.title('Energy Consumption vs. Load Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Node Utilization vs Load
    plt.subplot(2, 3, 6)
    plt.plot(load_factors, results['transformer']['node_utilizations'], 'bo-', label='TE-DQN')
    plt.plot(load_factors, results['dqn']['node_utilizations'], 'ro-', label='FF-DQN')
    plt.xlabel('Load Factor')
    plt.ylabel('Node Utilization')
    plt.title('Node Utilization vs. Load Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'varying_load_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to JSON for further analysis
    with open(os.path.join(save_dir, 'varying_load_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Varying load evaluation results saved to {save_dir}")
    return results


def main():
    """Main function to run the evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate VEC task offloading models under high-load conditions')
    parser.add_argument('--transformer_model', type=str, required=True, 
                        help='Path to TE-DQN model file (.pt)')
    parser.add_argument('--dqn_model', type=str, required=True,
                        help='Path to FF-DQN model file (.pth)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to environment configuration file (.json)')
    parser.add_argument('--energy_csv', type=str, required=True,
                        help='Path to energy consumption data CSV')
    parser.add_argument('--load_factor', type=float, default=2.0,
                        help='Load multiplication factor for high-load testing (default: 2.0)')
    parser.add_argument('--output_dir', type=str, default='high_load_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--varying_load', action='store_true',
                        help='Run evaluation with varying load factors')
    parser.add_argument('--load_factors', type=str, default='1.0,2.5,5.0,10.0,20.0',
                        help='Comma-separated list of load factors for varying load test')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load environment configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
            env_config = config['environment']
    except Exception as e:
        print(f"Error loading configuration file: {str(e)}")
        return
    
    # Ensure energy CSV path is set
    env_config['energy_csv_path'] = args.energy_csv
    
    # Check if model files exist
    if not os.path.exists(args.transformer_model):
        print(f"Error: Transformer model not found at {args.transformer_model}")
        return
    
    if not os.path.exists(args.dqn_model):
        print(f"Error: DQN model not found at {args.dqn_model}")
        return
    
    print("\n" + "="*80)
    print(f"VEC High Load Evaluation")
    print("="*80)
    
    # Log basic information
    print(f"Transformer Model: {args.transformer_model}")
    print(f"DQN Model: {args.dqn_model}")
    print(f"Energy Data: {args.energy_csv}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Evaluation Episodes: {args.episodes}")
    print(f"Seed: {args.seed}")
    
    # Save evaluation parameters
    eval_params = vars(args)
    with open(os.path.join(args.output_dir, 'evaluation_params.json'), 'w') as f:
        json.dump(eval_params, f, indent=4)
    
    # If varying load evaluation is requested
    if args.varying_load:
        print("\nRunning varying load evaluation...")
        load_factors = [float(x) for x in args.load_factors.split(',')]
        varying_results = run_varying_load_evaluation(
            args.transformer_model,
            args.dqn_model,
            env_config,
            load_factors=load_factors,
            num_episodes=args.episodes,
            save_dir=args.output_dir
        )
        print("\nVarying load evaluation completed.")
    else:
        # Create high-load environment for single load factor evaluation
        print(f"\nRunning high-load evaluation with load factor {args.load_factor}x")
        high_load_env = create_high_load_env(env_config, args.load_factor)
        
        # Evaluate Transformer model
        print("\nEvaluating TE-DQN model...")
        transformer_state_size = get_state_size(high_load_env, "transformer")
        transformer_agent = create_agent("transformer", transformer_state_size, high_load_env.action_space.n)
        
        transformer_results = None
        if load_model(transformer_agent, args.transformer_model):
            transformer_results = evaluate_model(
                high_load_env, 
                transformer_agent, 
                num_episodes=args.episodes, 
                model_type="transformer"
            )
            
            # Save results
            with open(os.path.join(args.output_dir, 'transformer_results.json'), 'w') as f:
                json.dump(transformer_results, f, indent=4)
        
        # Close SUMO to release the connection
        high_load_env.close()
        
        # Create new environment for DQN evaluation
        high_load_env = create_high_load_env(env_config, args.load_factor)
        
        # Evaluate DQN model
        print("\nEvaluating FF-DQN model...")
        dqn_state_size = get_state_size(high_load_env, "dqn")
        dqn_agent = create_agent("dqn", dqn_state_size, high_load_env.action_space.n)
        
        dqn_results = None
        if load_model(dqn_agent, args.dqn_model):
            dqn_results = evaluate_model(
                high_load_env, 
                dqn_agent, 
                num_episodes=args.episodes, 
                model_type="dqn"
            )
            
            # Save results
            with open(os.path.join(args.output_dir, 'dqn_results.json'), 'w') as f:
                json.dump(dqn_results, f, indent=4)
        
        # Close SUMO again
        high_load_env.close()
        
        # Compare models if both were successfully evaluated
        if transformer_results and dqn_results:
            compare_models(transformer_results, dqn_results, args.output_dir)
    
    print("\nEvaluation completed. Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()