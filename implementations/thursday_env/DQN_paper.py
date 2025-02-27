import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import os
import json
from collections import deque
from datetime import datetime

# Import the MEC environment
from mec_environment import MECEnvironment


class DQNNetwork(nn.Module):
    """Deep Q-Network with extended architecture"""
    def __init__(self, state_size, action_size, device):
        super(DQNNetwork, self).__init__()
        
        self.device = device
        
        # Verify state size
        print(f"Network initialized with state_size: {state_size}")
        
        # Network layers with better scaling
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_size)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(512)
        self.ln3 = nn.LayerNorm(256)
        
        self.dropout = nn.Dropout(0.2)  # Dropout for regularization
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        # Move to device
        self.to(device)
        
    def forward(self, x):
        x = x.to(self.device)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.ln3(self.fc3(x)))
        return self.fc4(x)


class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for MEC task offloading"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # DQN hyperparameters
        self.gamma = 0.995  # discount factor
        self.epsilon = 1.0  # starting exploration rate
        self.epsilon_min = 0.05  # minimum exploration rate
        self.epsilon_decay_steps = 1500  # number of episodes to reach epsilon_min
        self.current_episode = 0  # track current episode for linear decay
        
        self.learning_rate = 0.0005
        self.batch_size = 256
        self.min_replay_size = 5000
        
        # Create Q-Networks (current and target)
        self.q_network = DQNNetwork(state_size, action_size, self.device)
        self.target_network = DQNNetwork(state_size, action_size, self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                  lr=self.learning_rate,
                                  weight_decay=1e-5)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(100000)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        # Update epsilon using linear decay
        if self.current_episode < self.epsilon_decay_steps:
            self.epsilon = 1.0 - (1.0 - self.epsilon_min) * (self.current_episode / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_min
            
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            q_values = self.q_network(state)
            return q_values.cpu().argmax().item()
    
    def train(self):
        """Train the agent using experience replay"""
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values with target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = F.huber_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network parameters"""
        self.target_network.load_state_dict(self.q_network.state_dict())


def flatten_observation(obs, num_mvs, num_edge_servers):
    """
    Flatten the observation dictionary for a specific MV index into a vector
    suitable for DQN input
    """
    # For single MV agent (we'll process one MV at a time)
    mv_idx = 0
    
    # Extract and flatten relevant features for this MV
    flattened = []
    
    # Task-related features
    flattened.append(obs['task_size'][mv_idx][0])
    flattened.append(obs['required_cpu_cycles'][mv_idx][0])
    flattened.append(obs['task_deadline'][mv_idx][0])
    
    # MV capabilities
    flattened.append(obs['mv_compute_capacity'][mv_idx][0])
    flattened.append(obs['mv_local_queue'][mv_idx][0])
    flattened.append(obs['mv_energy'][mv_idx][0])
    
    # ES information
    flattened.extend(obs['es_compute_capacity'].flatten())
    flattened.extend(obs['es_queue_length'].flatten())
    
    # Network conditions for this MV
    for es_idx in range(num_edge_servers):
        flattened.append(obs['channel_gain'][mv_idx][es_idx])
        flattened.append(obs['network_delay'][mv_idx][es_idx])
    
    # Recent history of ES queue lengths (summarized)
    queue_history = obs['es_queue_history']
    # Take the most recent few timesteps to reduce dimensionality
    recent_history = queue_history[-5:, :]
    # Calculate statistics (mean, max) of recent history for each ES
    for es_idx in range(num_edge_servers):
        flattened.append(np.mean(recent_history[:, es_idx]))
        flattened.append(np.max(recent_history[:, es_idx]))
    
    return np.array(flattened, dtype=np.float32)


def get_state_size(env):
    """Calculate the state size based on the observation space"""
    obs = env.reset()
    flattened = flatten_observation(obs, env.num_mvs, env.num_edge_servers)
    return len(flattened)


def train_mec_dqn(env_config):
    """Training loop for MEC environment with DQN"""
    # Environment setup
    env = MECEnvironment(num_mvs=env_config['num_mvs'], 
                         num_edge_servers=env_config['num_edge_servers'],
                         continuous_action=False,
                         difficulty=env_config['difficulty'])
    
    # Calculate state size
    state_size = get_state_size(env)
    action_size = env.num_edge_servers + 1  # Local + each edge server

    print(f"Calculated state size: {state_size}")
    print(f"Action size: {action_size}")
    
    # Initialize DQN agent
    agent = DQNAgent(state_size, action_size)
    
    # Training parameters
    num_episodes = env_config['num_episodes']
    max_steps = env_config['max_steps']
    target_update_frequency = 10
    eval_frequency = 50
    
    # Enhanced metrics tracking
    metrics = {
        'rewards': [],
        'losses': [],
        'epsilons': [],
        'avg_rewards': [],
        'task_completion_rates': [],
        'processing_times': [],
        'energy_consumption': []
    }
    
    for episode in range(num_episodes):
        obs = env.reset()
        
        # Print observation structure in first episode
        if episode == 0:
            print("\nObservation structure:")
            for key, value in obs.items():
                print(f"{key}: shape {value.shape}")
        
        # Flatten the observation for a specific MV
        state = flatten_observation(obs, env.num_mvs, env.num_edge_servers)
        
        episode_reward = 0
        episode_losses = []
        episode_completion_rates = []
        episode_processing_times = []
        episode_energy = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_obs, rewards, done, info = env.step(action)
            
            # Get reward for the specific MV we're training
            reward = rewards[0]  # First MV in our case
            
            # Flatten the next observation
            next_state = flatten_observation(next_obs, env.num_mvs, env.num_edge_servers)
            
            # Store transition and train
            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)
            
            # Track metrics
            episode_completion_rates.append(info['task_completion_rate'])
            episode_processing_times.append(info['avg_processing_time'])
            episode_energy.append(info['avg_energy_consumption'])
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update target network and episode counter
        if episode % target_update_frequency == 0:
            agent.update_target_network()
        agent.current_episode = episode
        
        # Store metrics
        metrics['rewards'].append(episode_reward)
        metrics['task_completion_rates'].append(np.mean(episode_completion_rates))
        metrics['processing_times'].append(np.mean(episode_processing_times))
        metrics['energy_consumption'].append(np.mean(episode_energy))
        
        if episode_losses:
            metrics['losses'].append(np.mean(episode_losses))
        metrics['epsilons'].append(agent.epsilon)
        
        # Calculate running average
        if episode >= eval_frequency:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            metrics['avg_rewards'].append(avg_reward)
        
        # Print progress
        if episode % eval_frequency == 0:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:]) if episode >= eval_frequency else np.mean(metrics['rewards'])
            avg_completion = np.mean(metrics['task_completion_rates'][-eval_frequency:]) if episode >= eval_frequency else np.mean(metrics['task_completion_rates'])
            avg_time = np.mean(metrics['processing_times'][-eval_frequency:]) if episode >= eval_frequency else np.mean(metrics['processing_times'])
            avg_loss = np.mean(metrics['losses'][-eval_frequency:]) if metrics['losses'] else 0
            
            print(f"Episode {episode}/{num_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Completion Rate: {avg_completion:.2f}, "
                  f"Avg Time: {avg_time:.4f}, "
                  f"Avg Loss: {avg_loss:.4f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return agent, metrics


def save_training_results(metrics, env_config, save_dir="results"):
    """Save training metrics and plots to specified directory"""
    # Create results directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save configuration
    config_filename = os.path.join(save_dir, f"config_{timestamp}.json")
    with open(config_filename, 'w') as f:
        json.dump(env_config, f, indent=4)
    
    # Save metrics to JSON file
    metrics_filename = os.path.join(save_dir, f"metrics_{timestamp}.json")
    json_metrics = {
        'rewards': [float(r) for r in metrics['rewards']],
        'losses': [float(l) for l in metrics['losses']] if 'losses' in metrics else [],
        'epsilons': [float(e) for e in metrics['epsilons']],
        'avg_rewards': [float(ar) for ar in metrics.get('avg_rewards', [])],
        'task_completion_rates': [float(t) for t in metrics.get('task_completion_rates', [])],
        'processing_times': [float(p) for p in metrics.get('processing_times', [])],
        'energy_consumption': [float(e) for e in metrics.get('energy_consumption', [])]
    }
    with open(metrics_filename, 'w') as f:
        json.dump(json_metrics, f, indent=4)
    
    # Create and save plots
    # Plot 1: Rewards and Loss
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics['rewards'], alpha=0.6, label='Episode Reward')
    
    # Calculate and plot moving average
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
    plt.subplot(2, 2, 2)
    if 'losses' in metrics and metrics['losses']:
        plt.plot(metrics['losses'], label='Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
    
    # Plot task completion rate
    plt.subplot(2, 2, 3)
    if 'task_completion_rates' in metrics and metrics['task_completion_rates']:
        plt.plot(metrics['task_completion_rates'], label='Completion Rate')
        plt.xlabel('Episode')
        plt.ylabel('Completion Rate')
        plt.title('Task Completion Rate')
        plt.grid(True)
    
    # Plot processing time
    plt.subplot(2, 2, 4)
    if 'processing_times' in metrics and metrics['processing_times']:
        plt.plot(metrics['processing_times'], label='Processing Time')
        plt.xlabel('Episode')
        plt.ylabel('Time (s)')
        plt.title('Average Processing Time')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = os.path.join(save_dir, f"training_plot_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    # Second figure: Energy consumption and Epsilon
    plt.figure(figsize=(15, 5))
    
    # Plot energy consumption
    plt.subplot(1, 2, 1)
    if 'energy_consumption' in metrics and metrics['energy_consumption']:
        plt.plot(metrics['energy_consumption'], label='Energy')
        plt.xlabel('Episode')
        plt.ylabel('Energy (J)')
        plt.title('Average Energy Consumption')
        plt.grid(True)
    
    # Plot epsilon
    plt.subplot(1, 2, 2)
    plt.plot(metrics['epsilons'], label='Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate (Epsilon)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save second plot
    plot2_filename = os.path.join(save_dir, f"training_plot2_{timestamp}.png")
    plt.savefig(plot2_filename, dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print(f"Results saved to directory: {save_dir}")
    print(f"Config saved as: config_{timestamp}.json")
    print(f"Metrics saved as: metrics_{timestamp}.json")
    print(f"Plots saved as: training_plot_{timestamp}.png and training_plot2_{timestamp}.png")


if __name__ == "__main__":
    # Configuration
    env_config = {
        'num_mvs': 5,               # Number of mobile vehicles
        'num_edge_servers': 3,       # Number of edge servers
        'difficulty': 'normal',      # Difficulty level: 'easy', 'normal', or 'hard'
        'num_episodes': 2000,        # Number of training episodes
        'max_steps': 100             # Maximum steps per episode
    }
    
    # Create directory for this specific run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"run_{timestamp}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # Train the agent
    print(f"Starting training with configuration: {env_config}")
    agent, metrics = train_mec_dqn(env_config)
    
    # Save results
    save_training_results(metrics, env_config, save_dir=run_dir)
    
    print(f"Training completed. Results saved to {run_dir}")