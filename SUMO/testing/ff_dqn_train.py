import numpy as np
import os
import time
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import json

from vec_environment import VECEnvironment
from ff_dqn_agent import DQNAgent


def get_state_size(env):
    """Calculate the state size based on flattened observation space"""
    obs = env.reset()
    agent = DQNAgent(0, env.action_space.n)  # Temporary agent just for flattening
    flattened = agent.flatten_observation(obs)
    return len(flattened)


def train_dqn(env_config, agent_config, log_dir="results"):
    """Train DQN agent on the VEC environment"""
    # Create environment with enhanced parameters
    env = VECEnvironment(
        sumo_config=env_config['sumo_config'],
        simulation_duration=env_config['simulation_duration'],
        time_step=env_config['time_step'],
        queue_process_interval=env_config['queue_process_interval'],
        max_queue_length=env_config['max_queue_length'],
        history_length=env_config['history_length'],
        energy_csv_path=env_config.get('energy_csv_path'),
        energy_weight=env_config.get('energy_weight', 0.5),
        latency_model_params=env_config.get('latency_model_params'),
        min_tasks_per_step=env_config.get('min_tasks_per_step', 1),
        max_tasks_per_step=env_config.get('max_tasks_per_step', 5),
        task_generation_probability=env_config.get('task_generation_probability', 0.8),
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
    energy_consumptions = []
    idle_energies = []  # Track idle energy consumption
    avg_data_rates = []  # Track data rates for transmission
    node_usages = []  # Track node usage statistics
    
    # Training loop
    best_avg_reward = -float('inf')
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        state = agent.flatten_observation(state)
        
        episode_reward = 0
        step = 0
        total_energy = 0
        
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
            
            # Track energy consumption
            if 'energy_consumption' in info:
                total_energy += info['energy_consumption']
            
            # Log metrics
            if info.get('task_completion_rate') is not None:
                completion_rates.append(info['task_completion_rate'])
            if info.get('task_rejection_rate') is not None:
                rejection_rates.append(info['task_rejection_rate'])
            if info.get('task_drop_rate') is not None:
                drop_rates.append(info['task_drop_rate'])
            if info.get('avg_latency') is not None:
                avg_latencies.append(info['avg_latency'])
            if info.get('avg_energy_consumption') is not None:
                energy_consumptions.append(info['avg_energy_consumption'])
            if info.get('idle_energy') is not None:
                idle_energies.append(info['idle_energy'])
            if info.get('avg_data_rate') is not None:
                avg_data_rates.append(info['avg_data_rate'])
            
            # Calculate node usage
            if 'simulation_step' in info:
                node_usage = 0
                node_counts = 0
                for bs_id, bs_instance in env.base_station_instances.items():
                    active_count = sum(1 for node in bs_instance.nodes if node.active)
                    node_usage += active_count / len(bs_instance.nodes)
                    node_counts += 1
                if node_counts > 0:
                    node_usages.append(node_usage / node_counts)
            
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
        
        # Print progress with elapsed time
        if episode % 10 == 0:
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print(f"Episode {episode}/{num_episodes} [{int(hours)}h {int(minutes)}m {int(seconds)}s] - " +
                  f"Reward: {episode_reward:.2f}, " +
                  f"Avg Reward (100 ep): {running_avg:.2f}, " +
                  f"Epsilon: {agent.epsilon:.3f}, " +
                  f"Completion Rate: {info.get('task_completion_rate', 0):.3f}, " +
                  f"Avg Latency: {info.get('avg_latency', 0):.3f}, " +
                  f"Energy: {total_energy:.2f}")
        
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
        'avg_latencies': avg_latencies,
        'energy_consumptions': energy_consumptions,
        'idle_energies': idle_energies,
        'avg_data_rates': avg_data_rates,
        'node_usages': node_usages
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
    """Plot and save training curves with enhanced metrics"""
    plt.figure(figsize=(20, 15))
    
    # Plot rewards
    plt.subplot(3, 3, 1)
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
    plt.subplot(3, 3, 2)
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
    plt.subplot(3, 3, 3)
    if metrics['completion_rates']:
        plt.plot(metrics['completion_rates'], label='Completion Rate')
        plt.xlabel('Steps')
        plt.ylabel('Completion Rate')
        plt.title('Task Completion Rate')
        plt.grid(True)
    
    # Plot epsilon
    plt.subplot(3, 3, 4)
    plt.plot(metrics['epsilons'], label='Epsilon')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate (Epsilon)')
    plt.grid(True)
    
    # Plot average latency
    plt.subplot(3, 3, 5)
    if metrics['avg_latencies']:
        plt.plot(metrics['avg_latencies'], label='Avg Latency')
        plt.xlabel('Steps')
        plt.ylabel('Latency (s)')
        plt.title('Average Task Latency')
        plt.grid(True)
    
    # Plot task rejection rate
    plt.subplot(3, 3, 6)
    if metrics['rejection_rates']:
        plt.plot(metrics['rejection_rates'], label='Task Rejection Rate')
        plt.xlabel('Steps')
        plt.ylabel('Rejection Rate')
        plt.title('Task Rejection Rate')
        plt.grid(True)
        
    # Plot energy consumption
    plt.subplot(3, 3, 7)
    if metrics.get('energy_consumptions'):
        plt.plot(metrics['energy_consumptions'], label='Energy Consumption')
        plt.xlabel('Steps')
        plt.ylabel('Energy (J)')
        plt.title('Average Energy Consumption')
        plt.grid(True)
        
    # Plot idle energy consumption
    plt.subplot(3, 3, 8)
    if metrics.get('idle_energies'):
        plt.plot(metrics['idle_energies'], label='Idle Energy')
        plt.xlabel('Steps')
        plt.ylabel('Energy (J)')
        plt.title('Idle Energy Consumption')
        plt.grid(True)
        
    # Plot node usage
    plt.subplot(3, 3, 9)
    if metrics.get('node_usages'):
        plt.plot(metrics['node_usages'], label='Node Usage')
        plt.xlabel('Steps')
        plt.ylabel('Active Node Percentage')
        plt.title('Average Node Usage')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    
    # Create a second figure for more metrics
    if 'avg_data_rates' in metrics and metrics['avg_data_rates']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['avg_data_rates'], label='Avg Data Rate')
        plt.xlabel('Steps')
        plt.ylabel('Data Rate (Mbps)')
        plt.title('Average Transmission Data Rate')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'data_rate_curves.png'), dpi=300)
    
    plt.close('all')


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
    parser.add_argument('--energy_csv_path', type=str, default=None,
                        help='Path to energy consumption data CSV (optional)')
    parser.add_argument('--energy_weight', type=float, default=0.5,
                        help='Weight for energy consumption in reward (0.0-1.0)')
    parser.add_argument('--latency_freq', type=float, default=2.4,
                        help='Latency model frequency band (GHz)')
    parser.add_argument('--min_tasks', type=int, default=1,
                        help='Minimum tasks per step')
    parser.add_argument('--max_tasks', type=int, default=5,
                        help='Maximum tasks per step')
    parser.add_argument('--task_prob', type=float, default=0.8,
                        help='Task generation probability')
    
    args = parser.parse_args()
    
    # Environment configuration with enhanced parameters
    env_config = {
        'sumo_config': args.sumo_config,
        'simulation_duration': args.simulation_duration,
        'time_step': 1,
        'queue_process_interval': 5,
        'max_queue_length': 50,
        'history_length': 10,
        'energy_csv_path': args.energy_csv_path,
        'energy_weight': args.energy_weight,
        'latency_model_params': {
            'frequency_band': args.latency_freq,
            'bandwidth': 20,
            'noise_floor': -95
        },
        'min_tasks_per_step': args.min_tasks,
        'max_tasks_per_step': args.max_tasks,
        'task_generation_probability': args.task_prob,
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