import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import random

from vec_environment import VECEnvironment
from te_dqn_agent import VECTransformerAgent, flatten_observation, get_state_size

def train_transformer(env_config, agent_config, log_dir="transformer_results"):
    """Train Transformer agent on the VEC environment"""
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
    
    # Create transformer agent
    agent = VECTransformerAgent(state_size, action_size)
    
    # Training parameters
    num_episodes = agent_config['num_episodes']
    max_steps = agent_config['max_steps']
    eval_frequency = agent_config.get('eval_frequency', 50)
    
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
    
    # Metrics tracking
    metrics = {
        'rewards': [],
        'losses': [],
        'epsilons': [],
        'avg_rewards': [],
        'task_completion_rates': [],
        'rejection_rates': [],
        'drop_rates': [],
        'latencies': [],
        'node_usage': [],
        'energy_consumptions': [],
        'idle_energies': [],
        'avg_data_rates': []
    }
    
    # Pre-fill replay buffer with sequences from random episodes
    print("Pre-filling replay buffer with random experiences...")
    episode_count = 0
    sequence_count = 0
    
    while len(agent.replay_buffer) < agent.min_replay_size:
        episode_count += 1
        obs = env.reset()
        state = flatten_observation(obs)
        
        # Clear state history at the start of each episode
        agent.state_history.clear()
        
        for step in range(max_steps):
            # Select random action
            action = random.randrange(action_size)
            
            # Take action
            next_obs, reward, done, info = env.step(action)
            next_state = flatten_observation(next_obs)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update sequence count
            if len(agent.replay_buffer) > sequence_count:
                sequence_count = len(agent.replay_buffer)
                # Print progress occasionally
                if sequence_count % 100 == 0:
                    print(f"Collecting experiences: {sequence_count}/{agent.min_replay_size} sequences (episode {episode_count})")
            
            state = next_state
            
            if done:
                break
                
        # Force episode break after max_steps
        if step == max_steps - 1 and not done:
            # Add final transition with done=True to complete the episode
            agent.store_transition(state, action, reward, next_state, True)
    
    print(f"Replay buffer filled with {len(agent.replay_buffer)} sequences from {episode_count} episodes. Starting training...")
    
    # Training loop
    best_avg_reward = -float('inf')
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        state = flatten_observation(obs)
        
        episode_reward = 0
        episode_losses = []
        episode_completion_rates = []
        episode_rejection_rates = []
        episode_drop_rates = []
        episode_latencies = []
        episode_node_usage = []
        episode_energy_consumptions = []
        episode_idle_energies = []
        episode_data_rates = []
        total_energy = 0
        
        # Clear state history at the start of each episode
        agent.state_history.clear()
        
        step = 0
        while step < max_steps:
            # Select and perform action
            action = agent.select_action(state)
            next_obs, reward, done, info = env.step(action)
            next_state = flatten_observation(next_obs)
            
            # Store transition and train
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train()
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            step += 1
            
            # Track metrics
            if loss is not None:
                episode_losses.append(loss)
                
            # Track energy consumption
            if 'energy_consumption' in info:
                total_energy += info['energy_consumption']
                episode_energy_consumptions.append(info['energy_consumption'])
            
            # Log metrics from environment
            if info.get('task_completion_rate') is not None:
                episode_completion_rates.append(info['task_completion_rate'])
            if info.get('task_rejection_rate') is not None:
                episode_rejection_rates.append(info['task_rejection_rate'])
            if info.get('task_drop_rate') is not None:
                episode_drop_rates.append(info['task_drop_rate'])
            if info.get('avg_latency') is not None:
                episode_latencies.append(info['avg_latency'])
            if info.get('avg_energy_consumption') is not None:
                episode_energy_consumptions.append(info['avg_energy_consumption'])
            if info.get('idle_energy') is not None:
                episode_idle_energies.append(info['idle_energy'])
            if info.get('avg_data_rate') is not None:
                episode_data_rates.append(info['avg_data_rate'])
            
            # Calculate node usage - the average percentage of active nodes
            node_usage = 0
            node_counts = 0
            for bs_id, bs_instance in env.base_station_instances.items():
                active_count = sum(1 for node in bs_instance.nodes if node.active)
                node_usage += active_count / len(bs_instance.nodes)
                node_counts += 1
            if node_counts > 0:
                episode_node_usage.append(node_usage / node_counts)
            
            if done:
                break
        
        # Step the learning rate scheduler every 5 episodes
        if episode % 5 == 0:
            agent.scheduler.step()
        
        # Store metrics
        metrics['rewards'].append(episode_reward)
        agent.rewards.append(episode_reward)
        metrics['epsilons'].append(agent.epsilon)
        agent.epsilons.append(agent.epsilon)
        
        if episode_losses:
            metrics['losses'].append(np.mean(episode_losses))
            
        if episode_completion_rates:
            metrics['task_completion_rates'].append(np.mean(episode_completion_rates))
            
        if episode_rejection_rates:
            metrics['rejection_rates'].append(np.mean(episode_rejection_rates))
            
        if episode_drop_rates:
            metrics['drop_rates'].append(np.mean(episode_drop_rates))
            
        if episode_latencies:
            metrics['latencies'].append(np.mean(episode_latencies))
            
        if episode_node_usage:
            metrics['node_usage'].append(np.mean(episode_node_usage))
            
        if episode_energy_consumptions:
            metrics['energy_consumptions'].append(np.mean(episode_energy_consumptions))
            
        if episode_idle_energies:
            metrics['idle_energies'].append(np.mean(episode_idle_energies))
            
        if episode_data_rates:
            metrics['avg_data_rates'].append(np.mean(episode_data_rates))
        
        # Calculate running average
        running_avg = np.mean(metrics['rewards'][-min(episode, 100):])
        metrics['avg_rewards'].append(running_avg)
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
            model_path = os.path.join(run_dir, "best_model.pt")
            agent.save_model(model_path)
            print(f"Best model saved with avg reward: {best_avg_reward:.2f}")
        
        # Periodically save model
        if episode % 100 == 0:
            model_path = os.path.join(run_dir, f"model_episode_{episode}.pt")
            agent.save_model(model_path)
    
    # Save final model
    final_model_path = os.path.join(run_dir, "final_model.pt")
    agent.save_model(final_model_path)
    
    # Convert numpy arrays to lists for JSON serialization
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics[key] = value.tolist()
        elif isinstance(value, list) and value and isinstance(value[0], np.number):
            metrics[key] = [float(item) for item in value]
    
    # Save metrics
    metrics_path = os.path.join(run_dir, "transformer_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot and save training curves
    plot_training_curves(metrics, run_dir)
    
    return agent, metrics, run_dir

def plot_training_curves(metrics, save_dir):
    """Plot and save training curves with enhanced metrics"""
    plt.figure(figsize=(20, 15))
    
    # Plot rewards
    plt.subplot(3, 3, 1)
    plt.plot(metrics['rewards'], alpha=0.6, label='Episode Reward')
    window_size = 10
    if len(metrics['rewards']) > window_size:
        moving_avg = np.convolve(metrics['rewards'], 
                               np.ones(window_size)/window_size, 
                               mode='valid')
        x_avg = np.arange(window_size-1, len(metrics['rewards']))
        plt.plot(x_avg, moving_avg, 'r-', label='Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Training Rewards')
    plt.grid(True)
    
    # Plot losses
    plt.subplot(3, 3, 2)
    if 'losses' in metrics and metrics['losses']:
        plt.plot(metrics['losses'], label='Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
    
    # Plot task completion rate
    plt.subplot(3, 3, 3)
    if 'task_completion_rates' in metrics and metrics['task_completion_rates']:
        plt.plot(metrics['task_completion_rates'], label='Completion Rate')
        plt.xlabel('Episode')
        plt.ylabel('Completion Rate')
        plt.title('Task Completion Rate')
        plt.grid(True)
    
    # Plot epsilon
    plt.subplot(3, 3, 4)
    if 'epsilons' in metrics and metrics['epsilons']:
        plt.plot(metrics['epsilons'], label='Epsilon')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate (Epsilon)')
        plt.grid(True)
    
    # Plot average latency
    plt.subplot(3, 3, 5)
    if 'latencies' in metrics and metrics['latencies']:
        plt.plot(metrics['latencies'], label='Avg Latency')
        plt.xlabel('Episode')
        plt.ylabel('Time (s)')
        plt.title('Average Task Latency')
        plt.grid(True)
    
    # Plot energy consumption
    plt.subplot(3, 3, 6)
    if 'energy_consumptions' in metrics and metrics['energy_consumptions']:
        plt.plot(metrics['energy_consumptions'], label='Energy Consumption')
        plt.xlabel('Episode')
        plt.ylabel('Energy (J)')
        plt.title('Average Energy Consumption')
        plt.grid(True)
    
    # Plot node usage
    plt.subplot(3, 3, 7)
    if 'node_usage' in metrics and metrics['node_usage']:
        plt.plot(metrics['node_usage'], label='Node Usage')
        plt.xlabel('Episode')
        plt.ylabel('Active Node Percentage')
        plt.title('Average Node Usage')
        plt.grid(True)
    
    # Plot idle energy consumption
    plt.subplot(3, 3, 8)
    if 'idle_energies' in metrics and metrics['idle_energies']:
        plt.plot(metrics['idle_energies'], label='Idle Energy')
        plt.xlabel('Episode')
        plt.ylabel('Energy (J)')
        plt.title('Idle Energy Consumption')
        plt.grid(True)
    
    # Plot data rates
    plt.subplot(3, 3, 9)
    if 'avg_data_rates' in metrics and metrics['avg_data_rates']:
        plt.plot(metrics['avg_data_rates'], label='Avg Data Rate')
        plt.xlabel('Episode')
        plt.ylabel('Data Rate (Mbps)')
        plt.title('Average Transmission Data Rate')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = os.path.join(save_dir, "transformer_training_plot.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    # Create a second figure for rejection rates
    plt.figure(figsize=(10, 6))
    if 'rejection_rates' in metrics and metrics['rejection_rates']:
        plt.plot(metrics['rejection_rates'], label='Rejection Rate')
        plt.xlabel('Episode')
        plt.ylabel('Rejection Rate')
        plt.title('Task Rejection Rate')
        plt.grid(True)
        plt.tight_layout()
        
        # Save second plot
        plot2_filename = os.path.join(save_dir, "transformer_rejection_rates.png")
        plt.savefig(plot2_filename, dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    print(f"Training plots saved to {save_dir}")