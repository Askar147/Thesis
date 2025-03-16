import numpy as np
import os
import time
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import json

from vec_environment import VECEnvironment
from vec_dqn_agent import DQNAgent


def get_state_size(env):
    """Calculate the state size based on flattened observation space"""
    obs = env.reset()
    agent = DQNAgent(0, env.action_space.n)  # Temporary agent just for flattening
    flattened = agent.flatten_observation(obs)
    return len(flattened)


def train_dqn(env_config, agent_config, log_dir="results"):
    """Train DQN agent on the VEC environment"""
    # Create environment
    env = VECEnvironment(
        sumo_config=env_config['sumo_config'],
        simulation_duration=env_config['simulation_duration'],
        time_step=env_config['time_step'],
        queue_process_interval=env_config['queue_process_interval'],
        max_queue_length=env_config['max_queue_length'],
        history_length=env_config['history_length'],
        seed=env_config['seed']
    )
    
    # Calculate state size and create agent
    state_size = get_state_size(env)
    action_size = env.action_space.n
    print(f"State size: {state_size}, Action size: {action_size}")
    
    agent = DQNAgent(state_size, action_size)
    
    # Training parameters
    num_episodes = agent_config['num_episodes']
    max_steps = agent_config['max_steps']
    
    # Load existing model if specified
    if 'load_model' in agent_config and agent_config['load_model']:
        load_path = agent_config['load_model']
        agent.load_model(load_path)
    
    # Create directory for logging
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, f"run_{timestamp}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # Save configurations
    with open(os.path.join(run_dir, "env_config.json"), 'w') as f:
        json.dump(env_config, f, indent=4)
    with open(os.path.join(run_dir, "agent_config.json"), 'w') as f:
        json.dump(agent_config, f, indent=4)
    
    # Training metrics
    all_rewards = []
    episode_rewards = []
    completion_rates = []
    rejection_rates = []
    drop_rates = []
    avg_latencies = []
    
    # Training loop
    best_avg_reward = -float('inf')
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        state = agent.flatten_observation(state)
        
        episode_reward = 0
        step = 0
        
        while step < max_steps:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = agent.flatten_observation(next_state)
            
            # Store transition and train
            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.train()
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            step += 1
            
            # Log metrics
            if info.get('task_completion_rate') is not None:
                completion_rates.append(info['task_completion_rate'])
            if info.get('task_rejection_rate') is not None:
                rejection_rates.append(info['task_rejection_rate'])
            if info.get('task_drop_rate') is not None:
                drop_rates.append(info['task_drop_rate'])
            if info.get('avg_latency') is not None:
                avg_latencies.append(info['avg_latency'])
            
            if done:
                break
        
        # Update epsilon and target network
        agent.update_epsilon()
        
        if episode % agent_config['target_update_frequency'] == 0:
            agent.update_target_network()
        
        # Log episode metrics
        all_rewards.append(episode_reward)
        episode_rewards.append(episode_reward)
        agent.rewards.append(episode_reward)
        agent.episodes += 1
        
        # Calculate and log running average
        running_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        agent.avg_rewards.append(running_avg)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} - " +
                  f"Reward: {episode_reward:.2f}, " +
                  f"Avg Reward (100 ep): {running_avg:.2f}, " +
                  f"Epsilon: {agent.epsilon:.3f}, " +
                  f"Completion Rate: {info.get('task_completion_rate', 0):.3f}, " +
                  f"Avg Latency: {info.get('avg_latency', 0):.3f}")
        
        # Save model if it's the best so far
        if running_avg > best_avg_reward and episode > 100:
            best_avg_reward = running_avg
            model_path = os.path.join(run_dir, "best_model.pth")
            agent.save_model(model_path)
            print(f"Best model saved with avg reward: {best_avg_reward:.2f}")
        
        # Periodically save model
        if episode % 100 == 0:
            model_path = os.path.join(run_dir, f"model_episode_{episode}.pth")
            agent.save_model(model_path)
    
    # Save final model
    final_model_path = os.path.join(run_dir, "final_model.pth")
    agent.save_model(final_model_path)
    
    # Save metrics
    metrics = {
        'rewards': all_rewards,
        'episode_rewards': episode_rewards,
        'avg_rewards': agent.avg_rewards,
        'losses': agent.losses,
        'epsilons': agent.epsilons,
        'completion_rates': completion_rates,
        'rejection_rates': rejection_rates,
        'drop_rates': drop_rates,
        'avg_latencies': avg_latencies
    }
    
    # Convert numpy arrays to lists for JSON serialization
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics[key] = value.tolist()
        elif isinstance(value, list) and value and isinstance(value[0], np.number):
            metrics[key] = [float(item) for item in value]
    
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    # Plot and save training curves
    plot_training_curves(metrics, run_dir)
    
    return agent, metrics, run_dir


def plot_training_curves(metrics, save_dir):
    """Plot and save training curves"""
    plt.figure(figsize=(20, 12))
    
    # Plot rewards
    plt.subplot(2, 3, 1)
    plt.plot(metrics['rewards'], alpha=0.3, label='Episode Reward')
    window_size = 100
    if len(metrics['rewards']) > window_size:
        moving_avg = np.convolve(metrics['rewards'], 
                               np.ones(window_size)/window_size, 
                               mode='valid')
        x_avg = np.arange(window_size-1, len(metrics['rewards']))
        plt.plot(x_avg, moving_avg, 'r-', label='Moving Average (100 ep)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    
    # Plot losses
    plt.subplot(2, 3, 2)
    if metrics['losses']:
        # Sample losses if there are too many
        sample_rate = max(1, len(metrics['losses']) // 1000)
        losses = metrics['losses'][::sample_rate]
        plt.plot(np.arange(0, len(metrics['losses']), sample_rate), losses, label='Loss')
        plt.xlabel('Training steps (sampled)')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
    
    # Plot task completion rate
    plt.subplot(2, 3, 3)
    if metrics['completion_rates']:
        plt.plot(metrics['completion_rates'], label='Completion Rate')
        plt.xlabel('Steps')
        plt.ylabel('Completion Rate')
        plt.title('Task Completion Rate')
        plt.grid(True)
    
    # Plot epsilon
    plt.subplot(2, 3, 4)
    plt.plot(metrics['epsilons'], label='Epsilon')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate (Epsilon)')
    plt.grid(True)
    
    # Plot average latency
    plt.subplot(2, 3, 5)
    if metrics['avg_latencies']:
        plt.plot(metrics['avg_latencies'], label='Avg Latency')
        plt.xlabel('Steps')
        plt.ylabel('Latency (s)')
        plt.title('Average Task Latency')
        plt.grid(True)
    
    # Plot task rejection rate
    plt.subplot(2, 3, 6)
    if metrics['rejection_rates']:
        plt.plot(metrics['rejection_rates'], label='Task Rejection Rate')
        plt.xlabel('Steps')
        plt.ylabel('Rejection Rate')
        plt.title('Task Rejection Rate')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN agent for VEC task offloading')
    parser.add_argument('--sumo_config', type=str, default='astana.sumocfg', 
                        help='Path to SUMO configuration file')
    parser.add_argument('--simulation_duration', type=int, default=300,
                        help='Total simulation duration (seconds)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load existing model (optional)')
    parser.add_argument('--log_dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Environment configuration
    env_config = {
        'sumo_config': args.sumo_config,
        'simulation_duration': args.simulation_duration,
        'time_step': 1,
        'queue_process_interval': 5,
        'max_queue_length': 50,
        'history_length': 10,
        'seed': args.seed
    }
    
    # Agent configuration
    agent_config = {
        'num_episodes': args.episodes,
        'max_steps': args.max_steps,
        'target_update_frequency': 10,
        'load_model': args.load_model
    }
    
    # Train agent
    start_time = time.time()
    agent, metrics, run_dir = train_dqn(env_config, agent_config, args.log_dir)
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Results saved to {run_dir}")