#!/usr/bin/env python3
"""
Evaluation Script for TE-DDQN (Transformer) models in VEC environment
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict, deque
import traceback
import time
import math

from vec_environment_2 import VECEnvironment

class PositionalEncoding(torch.nn.Module):
    """Positional encoding for temporal information"""
    def __init__(self, d_model, max_seq_length=100):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class VECTransformer(torch.nn.Module):
    """Transformer for VEC task offloading decisions"""
    def __init__(self, state_dim, action_dim, seq_length=16, d_model=128, nhead=4, 
                 num_layers=3, dropout=0.1, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_length = seq_length
        self.d_model = d_model
        self.device = device
        
        # Input normalization
        self.state_norm = torch.nn.LayerNorm(state_dim)
        
        # Input projection with layer normalization
        self.input_projection = torch.nn.Linear(state_dim, d_model)
        self.input_norm = torch.nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, seq_length)
        
        # Transformer encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Dueling Network Architecture
        # Value stream
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.LayerNorm(d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.LayerNorm(d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model // 2, 1)
        )
        
        # Advantage stream
        self.advantage_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.LayerNorm(d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.LayerNorm(d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model // 2, action_dim)
        )
        
        # Move model to device
        self.to(device)
        
    def forward(self, states):
        # Handle single state vs. batch of states
        if len(states.shape) == 2:
            # Single sequence - add batch dimension
            states = states.unsqueeze(0)
            
        # Normalize input states
        x = self.state_norm(states)
        
        # Project to d_model dimension and apply normalization
        x = self.input_projection(x)
        x = self.input_norm(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Get last sequence element for predictions
        x = x[:, -1]
        
        # Dueling architecture
        value = self.value_head(x)
        advantage = self.advantage_head(x)
        
        # Combine value and advantage for Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values, value

class VECTransformerAgent:
    """Transformer-based agent for VEC task offloading"""
    def __init__(self, state_dim, action_dim, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        print(f"Using device: {self.device}")
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        
        # Hyperparameters
        self.seq_length = 16  # Sequence length for transformer
        self.gamma = 0.99     # Discount factor
        self.tau = 0.01       # Soft update parameter
        
        # Exploration parameters - not used in evaluation
        self.epsilon = 0.0
        
        # Initialize networks
        self.policy_net = VECTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            seq_length=self.seq_length,
            device=self.device
        )
        
        self.target_net = VECTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            seq_length=self.seq_length,
            device=self.device
        )
        
        # Initialize optimizer (needed for loading model)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.0003, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=400, gamma=0.5)
        
        # State history for sequence building
        self.state_history = deque(maxlen=self.seq_length)
        
    def select_action(self, state, evaluate=True):
        """Select action using policy network"""
        # Add state to history
        self.state_history.append(state)
        
        # Pad history if needed
        if len(self.state_history) < self.seq_length:
            padding = [state] * (self.seq_length - len(self.state_history))
            seq_states = padding + list(self.state_history)
        else:
            seq_states = list(self.state_history)
        
        # Convert to tensor
        seq_tensor = torch.FloatTensor(np.array(seq_states)).to(self.device)
        
        # Get action from network
        with torch.no_grad():
            q_values, _ = self.policy_net(seq_tensor)
            return q_values.argmax().item()
    
    def load_model(self, path):
        """Load model weights"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Transformer model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading transformer model: {e}")
            return False

def flatten_observation(obs):
    """
    Flatten a dictionary observation from VEC environment into a vector
    suitable for the Transformer input
    """
    flattened = []
    
    # Add task information
    flattened.append(obs['task_size'][0])
    flattened.append(obs['required_cpu_cycles'][0])
    flattened.append(obs['task_deadline'][0])
    flattened.append(obs['task_scenario_id'][0]) 
    
    # Add vehicle information
    flattened.append(obs['vehicle_pos_x'][0])
    flattened.append(obs['vehicle_pos_y'][0])
    flattened.append(obs['vehicle_speed'][0])
    
    # Add base station information
    flattened.append(obs['distance_to_bs'][0])
    flattened.append(obs['bs_queue_length'][0])
    
    # Add edge node information
    flattened.append(obs['active_nodes'][0])
    
    # Add node load information
    flattened.extend(obs['node_loads'])
    
    # Add node active status
    flattened.extend(obs['node_active_status'])
    
    # Add historical load information (flattened)
    flattened.extend(obs['historical_loads'].flatten())
    
    return np.array(flattened, dtype=np.float32)

def get_state_size(env):
    """Calculate the state size based on flattened observation space"""
    obs = env.reset()
    flattened = flatten_observation(obs)
    return len(flattened)

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
        
        # Clear state history at the start of each episode
        agent.state_history.clear()
        
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
    """Analyze Transformer model's performance across different scenarios"""
    print(f"\nAnalyzing Transformer model from {model_path}")
    
    # Create result directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"transformer_eval_{timestamp}")
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
    agent = VECTransformerAgent(state_size, action_size)
    
    # Load model
    success = agent.load_model(model_path)
    if not success:
        print(f"Failed to load model from {model_path}")
        return None
    
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
    
    print(f"Analysis completed for Transformer model")
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
    
    plt.suptitle(f"Transformer Model Performance Across Scenarios", fontsize=16)
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
    parser = argparse.ArgumentParser(description='Evaluate TE-DDQN (Transformer) model in VEC environment')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to the trained Transformer model checkpoint')
    parser.add_argument('--output_dir', type=str, default='transformer_evaluation',
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
        
        # Analyze Transformer model
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