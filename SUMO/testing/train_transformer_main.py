#!/usr/bin/env python3
"""
Main script to run the Transformer-based agent for VEC task offloading
with the enhanced environment
"""

import os
import argparse
import json
from datetime import datetime

from vec_environment import VECEnvironment
from te_dqn_agent import VECTransformerAgent, get_state_size, flatten_observation
from te_dqn_train import train_transformer
from te_dqn_train import plot_training_curves

def main():
    """Main function to run Transformer training with enhanced environment"""
    parser = argparse.ArgumentParser(description='Train Transformer agent with enhanced VEC environment')
    parser.add_argument('--sumo_config', type=str, default='astana.sumocfg', 
                        help='Path to SUMO configuration file')
    parser.add_argument('--energy_csv', type=str, required=True,
                        help='Path to energy consumption data CSV (required)')
    parser.add_argument('--energy_weight', type=float, default=0.5,
                        help='Weight for energy consumption in reward (0.0-1.0)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=5000,
                        help='Maximum steps per episode')
    parser.add_argument('--duration', type=int, default=300,
                        help='Simulation duration in seconds')
    parser.add_argument('--output_dir', type=str, default='transformer_results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--seq_length', type=int, default=16,
                        help='Sequence length for transformer')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension for transformer')
    parser.add_argument('--nheads', type=int, default=4,
                        help='Number of attention heads')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"transformer_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Environment configuration
    env_config = {
        'sumo_config': args.sumo_config,
        'simulation_duration': args.duration,
        'time_step': 1,
        'queue_process_interval': 5,
        'max_queue_length': 50,
        'history_length': 10,
        'energy_csv_path': args.energy_csv,
        'energy_weight': args.energy_weight,
        'latency_model_params': {
            'frequency_band': 2.4,
            'bandwidth': 20,
            'noise_floor': -95
        },
        'min_tasks_per_step': 10,
        'max_tasks_per_step': 20,
        'task_generation_probability': 1,
        'seed': args.seed
    }
    
    # Agent configuration
    agent_config = {
        'num_episodes': args.episodes,
        'max_steps': args.max_steps,
        'eval_frequency': 50,
        'seq_length': args.seq_length,
        'd_model': args.d_model,
        'nhead': args.nheads,
        'num_layers': 3,
        'dropout': 0.1,
        'batch_size': 64,
        'gamma': 0.99,
        'tau': 0.01,
        'min_replay_size': 1000,
        'epsilon_min': 0.05,
        'epsilon_decay_steps': 2000,
        'lr': 0.0003,
        'weight_decay': 1e-5,
        'load_model': None
    }
    
    # Save configuration
    with open(os.path.join(run_dir, 'transformer_config.json'), 'w') as f:
        json.dump({'environment': env_config, 'agent': agent_config}, f, indent=4)
    
    print(f"Starting Transformer training with enhanced environment...")
    print(f"Using energy data from: {args.energy_csv}")
    print(f"Energy weight: {args.energy_weight}")
    print(f"Episodes: {args.episodes}")
    print(f"Transformer sequence length: {args.seq_length}")
    print(f"Output directory: {run_dir}")
    
    # Run training
    agent, metrics, output_dir = train_transformer(env_config, agent_config, log_dir=run_dir)
    
    print(f"Training completed!")
    print(f"Results saved to: {output_dir}")
    
    # Optional: Evaluate the trained model
    print("Would you like to evaluate the trained model? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        print("Running evaluation of the best model...")
        # Create evaluation environment
        eval_env = VECEnvironment(
            sumo_config=env_config['sumo_config'],
            simulation_duration=env_config['simulation_duration'],
            time_step=env_config['time_step'],
            queue_process_interval=env_config['queue_process_interval'],
            max_queue_length=env_config['max_queue_length'],
            history_length=env_config['history_length'],
            energy_csv_path=env_config['energy_csv_path'],
            energy_weight=env_config['energy_weight'],
            latency_model_params=env_config['latency_model_params'],
            min_tasks_per_step=env_config['min_tasks_per_step'],
            max_tasks_per_step=env_config['max_tasks_per_step'],
            task_generation_probability=env_config['task_generation_probability'],
            seed=args.seed + 100  # Different seed for evaluation
        )
        
        # Calculate state size
        state_size = get_state_size(eval_env)
        action_size = eval_env.action_space.n
        
        # Create evaluation agent
        eval_agent = VECTransformerAgent(state_size, action_size)
        
        # Load the best model
        best_model_path = os.path.join(output_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            eval_agent.load_model(best_model_path)
            
            # Run 10 evaluation episodes
            eval_episodes = 10
            total_rewards = []
            
            for episode in range(eval_episodes):
                obs = eval_env.reset()
                state = flatten_observation(obs)
                episode_reward = 0
                
                # Clear state history
                eval_agent.state_history.clear()
                
                step = 0
                max_eval_steps = 300
                
                while step < max_eval_steps:
                    # Select action deterministically (no exploration)
                    action = eval_agent.select_action(state, evaluate=True)
                    
                    # Take action
                    next_obs, reward, done, info = eval_env.step(action)
                    next_state = flatten_observation(next_obs)
                    
                    episode_reward += reward
                    state = next_state
                    step += 1
                    
                    if done:
                        break
                
                total_rewards.append(episode_reward)
                print(f"Evaluation Episode {episode+1}/{eval_episodes} - Reward: {episode_reward:.2f}")
            
            # Print evaluation results
            avg_reward = sum(total_rewards) / len(total_rewards)
            print(f"\nEvaluation Results:")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Min Reward: {min(total_rewards):.2f}")
            print(f"Max Reward: {max(total_rewards):.2f}")
            
            # Save evaluation results
            eval_results = {
                'rewards': total_rewards,
                'avg_reward': avg_reward,
                'min_reward': min(total_rewards),
                'max_reward': max(total_rewards)
            }
            
            with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
                json.dump(eval_results, f, indent=4)
        else:
            print(f"Could not find best model at {best_model_path}")

if __name__ == "__main__":
    main()