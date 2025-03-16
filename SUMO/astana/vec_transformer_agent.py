import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import random
import matplotlib.pyplot as plt
import os
import json
import math
import time
from collections import deque
from datetime import datetime

# Import the VEC environment
from vec_environment import VECEnvironment

class PositionalEncoding(nn.Module):
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

class VECTransformer(nn.Module):
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
        self.state_norm = nn.LayerNorm(state_dim)
        
        # Input projection with layer normalization
        self.input_projection = nn.Linear(state_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Dueling Network Architecture
        # Value stream
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Advantage stream
        self.advantage_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_dim)
        )
        
        # Move model to device
        self.to(device)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights using appropriate initialization methods"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.kaiming_normal_(param)  # Better init for ReLU
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
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

class SequentialReplayBuffer:
    """Replay buffer that stores sequences of experiences"""
    def __init__(self, capacity, seq_length):
        self.buffer = deque(maxlen=capacity)
        self.seq_length = seq_length
        self.episode_buffer = []
        
    def push(self, state, action, reward, next_state, done):
        # Add to episode buffer
        self.episode_buffer.append((state, action, reward, next_state, done))
        
        # If we have enough transitions in the episode buffer, store a sequence
        if len(self.episode_buffer) >= self.seq_length:
            # Add the most recent sequence to the replay buffer
            sequence = self.episode_buffer[-self.seq_length:]
            self.buffer.append(sequence)
        
        # If episode is done, reset episode buffer
        if done:
            self.episode_buffer = []
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        # Sample random sequences
        batch = random.sample(self.buffer, batch_size)
        
        # Extract and organize data
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for sequence in batch:
            seq_states = []
            seq_actions = []
            seq_rewards = []
            seq_next_states = []
            seq_dones = []
            
            for transition in sequence:
                s, a, r, ns, d = transition
                seq_states.append(s)
                seq_actions.append(a)
                seq_rewards.append(r)
                seq_next_states.append(ns)
                seq_dones.append(d)
            
            states.append(seq_states)
            actions.append(seq_actions)
            rewards.append(seq_rewards)
            next_states.append(seq_next_states)
            dones.append(seq_dones)
        
        # Convert to tensors
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones))
        )
    
    def __len__(self):
        return len(self.buffer)

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
        self.batch_size = 64  # Batch size
        self.min_replay_size = 1000  # Minimum replay buffer size before learning
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay_steps = 2000
        self.current_step = 0
        
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
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003, weight_decay=1e-5)
        self.scheduler = StepLR(self.optimizer, step_size=400, gamma=0.5)  # Learning rate scheduler
        
        # Initialize replay buffer
        self.replay_buffer = SequentialReplayBuffer(100000, self.seq_length)
        
        # State history for sequence building
        self.state_history = deque(maxlen=self.seq_length)
        
        # Performance tracking
        self.update_count = 0
        self.recent_losses = deque(maxlen=100)
        
    def select_action(self, state, evaluate=False):
        """Select action using epsilon-greedy policy"""
        # Update epsilon with linear decay
        if self.current_step < self.epsilon_decay_steps:
            self.epsilon = 1.0 - (1.0 - self.epsilon_min) * (self.current_step / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_min
        
        # Random exploration
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
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
    
    def train(self):
        """Train the agent using experience replay"""
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        self.current_step += 1
        
        # Sample from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)
        
        # Get current Q values
        q_values, _ = self.policy_net(state_batch)
        current_q = q_values.gather(1, action_batch[:, -1].unsqueeze(1))
        
        # Get target Q values using double Q-learning
        with torch.no_grad():
            # Get actions from policy network
            next_q_values, _ = self.policy_net(next_state_batch)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            
            # Get Q-values from target network for those actions
            target_q_values, _ = self.target_net(next_state_batch)
            next_q = target_q_values.gather(1, next_actions)
            
            # Compute target Q values
            target_q = reward_batch[:, -1].unsqueeze(1) + \
                      (1 - done_batch[:, -1].unsqueeze(1)) * self.gamma * next_q
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Soft update target network
        if self.update_count % 10 == 0:
            for target_param, policy_param in zip(self.target_net.parameters(), 
                                                 self.policy_net.parameters()):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1 - self.tau) * target_param.data
                )
        
        self.update_count += 1
        self.recent_losses.append(loss.item())
        
        return loss.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def save_model(self, path):
        """Save model weights"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'step': self.current_step
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.current_step = checkpoint['step']
        print(f"Model loaded from {path}")

def train_vec_transformer(env_config):
    """Training loop for VEC environment with Transformer"""
    # Environment setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = VECEnvironment(
        sumo_config=env_config['sumo_config'],
        simulation_duration=env_config['simulation_duration'],
        time_step=env_config['time_step'],
        queue_process_interval=env_config['queue_process_interval'],
        max_queue_length=env_config['max_queue_length'],
        history_length=env_config['history_length'],
        seed=env_config['seed']
    )
    
    # Calculate state size
    state_size = get_state_size(env)
    action_size = env.action_space.n
    
    print(f"Calculated state size: {state_size}")
    print(f"Action size: {action_size}")
    
    # Initialize Transformer agent
    agent = VECTransformerAgent(state_size, action_size)
    
    # Training parameters
    num_episodes = env_config['num_episodes']
    max_steps = env_config['max_steps']
    eval_frequency = 50
    
    # Enhanced metrics tracking
    metrics = {
        'rewards': [],
        'losses': [],
        'epsilons': [],
        'avg_rewards': [],
        'task_completion_rates': [],
        'rejection_rates': [],
        'drop_rates': [],
        'latencies': [],
        'node_usage': []
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
    
    # Start training timer
    start_time = time.time()
    
    for episode in range(num_episodes):
        obs = env.reset()
        state = flatten_observation(obs)
        
        episode_reward = 0
        episode_losses = []
        episode_completion_rates = []
        episode_rejection_rates = []
        episode_drop_rates = []
        episode_latencies = []
        episode_node_usage = []
        
        # Clear state history at the start of each episode
        agent.state_history.clear()
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_obs, reward, done, info = env.step(action)
            next_state = flatten_observation(next_obs)
            
            # Store transition and train
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train()
            
            if loss is not None:
                episode_losses.append(loss)
            
            # Track metrics
            if 'task_completion_rate' in info:
                episode_completion_rates.append(info['task_completion_rate'])
            if 'task_rejection_rate' in info:
                episode_rejection_rates.append(info['task_rejection_rate'])
            if 'task_drop_rate' in info:
                episode_drop_rates.append(info['task_drop_rate'])
            if 'avg_latency' in info:
                episode_latencies.append(info['avg_latency'])
                
            # Calculate node usage - the average percentage of active nodes
            node_usage = 0
            node_counts = 0
            for bs_id, bs_instance in env.base_station_instances.items():
                active_count = sum(1 for node in bs_instance.nodes if node.active)
                node_usage += active_count / len(bs_instance.nodes)
                node_counts += 1
            if node_counts > 0:
                episode_node_usage.append(node_usage / node_counts)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Step the learning rate scheduler every 5 episodes
        if episode % 5 == 0:
            agent.scheduler.step()
        
        # Store metrics
        metrics['rewards'].append(episode_reward)
        metrics['epsilons'].append(agent.epsilon)
        
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
        
        # Calculate running average
        if episode >= eval_frequency:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            metrics['avg_rewards'].append(avg_reward)
        
        # Print progress
        if episode % eval_frequency == 0:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:]) if episode >= eval_frequency else np.mean(metrics['rewards'])
            avg_completion = np.mean(metrics['task_completion_rates'][-eval_frequency:]) if metrics['task_completion_rates'] else 0
            avg_latency = np.mean(metrics['latencies'][-eval_frequency:]) if metrics['latencies'] else 0
            avg_loss = np.mean(metrics['losses'][-eval_frequency:]) if metrics['losses'] else 0
            
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print(f"Episode {episode}/{num_episodes} [{int(hours)}h {int(minutes)}m {int(seconds)}s] | "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Completion Rate: {avg_completion:.2f}, "
                  f"Avg Latency: {avg_latency:.4f}, "
                  f"Avg Loss: {avg_loss:.4f}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"LR: {agent.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save model periodically
            if episode > 0 and episode % (eval_frequency * 5) == 0:
                model_dir = os.path.join("transformer_results", f"run_{timestamp}")
                os.makedirs(model_dir, exist_ok=True)
                agent.save_model(os.path.join(model_dir, f"transformer_model_episode_{episode}.pt"))
    
    # Save final model
    model_dir = os.path.join("transformer_results", f"run_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    agent.save_model(os.path.join(model_dir, "transformer_model_final.pt"))
    
    return agent, metrics

def evaluate_transformer_agent(agent, env, num_episodes=10):
    """Evaluate transformer agent performance without exploration"""
    total_rewards = []
    task_completion_rates = []
    rejection_rates = []
    drop_rates = []
    latencies = []
    node_usage = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        state = flatten_observation(obs)
        
        episode_reward = 0
        episode_completion_rates = []
        episode_rejection_rates = []
        episode_drop_rates = []
        episode_latencies = []
        episode_node_usage = []
        
        # Clear state history at the start of each episode
        agent.state_history.clear()
        
        max_steps = 300
        for step in range(max_steps):
            # Select action deterministically
            action = agent.select_action(state, evaluate=True)
            
            # Take action
            next_obs, reward, done, info = env.step(action)
            
            # Track metrics
            if 'task_completion_rate' in info:
                episode_completion_rates.append(info['task_completion_rate'])
            if 'task_rejection_rate' in info:
                episode_rejection_rates.append(info['task_rejection_rate'])
            if 'task_drop_rate' in info:
                episode_drop_rates.append(info['task_drop_rate'])
            if 'avg_latency' in info:
                episode_latencies.append(info['avg_latency'])
                
            # Calculate node usage
            node_usage_val = 0
            node_counts = 0
            for bs_id, bs_instance in env.base_station_instances.items():
                active_count = sum(1 for node in bs_instance.nodes if node.active)
                node_usage_val += active_count / len(bs_instance.nodes)
                node_counts += 1
            if node_counts > 0:
                episode_node_usage.append(node_usage_val / node_counts)
            
            # Update state and reward
            next_state = flatten_observation(next_obs)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        
        if episode_completion_rates:
            task_completion_rates.append(np.mean(episode_completion_rates))
        if episode_rejection_rates:
            rejection_rates.append(np.mean(episode_rejection_rates))
        if episode_drop_rates:
            drop_rates.append(np.mean(episode_drop_rates))
        if episode_latencies:
            latencies.append(np.mean(episode_latencies))
        if episode_node_usage:
            node_usage.append(np.mean(episode_node_usage))
    
    # Return average metrics
    return {
        'avg_reward': np.mean(total_rewards),
        'avg_completion_rate': np.mean(task_completion_rates) if task_completion_rates else 0,
        'avg_rejection_rate': np.mean(rejection_rates) if rejection_rates else 0,
        'avg_drop_rate': np.mean(drop_rates) if drop_rates else 0,
        'avg_latency': np.mean(latencies) if latencies else 0,
        'avg_node_usage': np.mean(node_usage) if node_usage else 0
    }

def save_transformer_training_results(metrics, env_config, save_dir="transformer_results"):
    """Save training metrics and plots to specified directory"""
    # Create results directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save configuration
    config_filename = os.path.join(save_dir, f"transformer_config_{timestamp}.json")
    with open(config_filename, 'w') as f:
        json.dump(env_config, f, indent=4)
    
    # Save metrics to JSON file
    metrics_filename = os.path.join(save_dir, f"transformer_metrics_{timestamp}.json")
    json_metrics = {}
    
    for key, value in metrics.items():
        if isinstance(value, list):
            if value and isinstance(value[0], np.ndarray):
                json_metrics[key] = [float(v) for v in value]
            elif value and isinstance(value[0], np.number):
                json_metrics[key] = [float(v) for v in value]
            else:
                json_metrics[key] = value
    
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
    
    # Plot latency
    plt.subplot(2, 2, 4)
    if 'latencies' in metrics and metrics['latencies']:
        plt.plot(metrics['latencies'], label='Latency')
        plt.xlabel('Episode')
        plt.ylabel('Time (s)')
        plt.title('Average Task Latency')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = os.path.join(save_dir, f"transformer_training_plot_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    # Second figure: Node usage, rejection rates, and epsilon
    plt.figure(figsize=(15, 5))
    
    # Plot node usage
    plt.subplot(1, 3, 1)
    if 'node_usage' in metrics and metrics['node_usage']:
        plt.plot(metrics['node_usage'], label='Node Usage')
        plt.xlabel('Episode')
        plt.ylabel('Active Node Percentage')
        plt.title('Average Node Usage')
        plt.grid(True)
    
    # Plot rejection rates
    plt.subplot(1, 3, 2)
    if 'rejection_rates' in metrics and metrics['rejection_rates']:
        plt.plot(metrics['rejection_rates'], label='Rejection Rate')
        plt.xlabel('Episode')
        plt.ylabel('Rejection Rate')
        plt.title('Task Rejection Rate')
        plt.grid(True)
    
    # Plot epsilon
    plt.subplot(1, 3, 3)
    plt.plot(metrics['epsilons'], label='Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate (Epsilon)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save second plot
    plot2_filename = os.path.join(save_dir, f"transformer_training_plot2_{timestamp}.png")
    plt.savefig(plot2_filename, dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print(f"Results saved to directory: {save_dir}")
    print(f"Config saved as: transformer_config_{timestamp}.json")
    print(f"Metrics saved as: transformer_metrics_{timestamp}.json")
    print(f"Plots saved as: transformer_training_plot_{timestamp}.png and transformer_training_plot2_{timestamp}.png")

def compare_models(dqn_results_dir, transformer_results_dir, save_dir="comparison_results"):
    """Compare DQN and Transformer model performances"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load DQN metrics
    dqn_metrics_file = None
    for file in os.listdir(dqn_results_dir):
        if file.startswith("metrics") and file.endswith(".json"):
            dqn_metrics_file = os.path.join(dqn_results_dir, file)
            break
    
    if not dqn_metrics_file:
        print("No DQN metrics found.")
        return
    
    with open(dqn_metrics_file, 'r') as f:
        dqn_metrics = json.load(f)
    
    # Load Transformer metrics
    transformer_metrics_file = None
    for file in os.listdir(transformer_results_dir):
        if file.startswith("transformer_metrics_") and file.endswith(".json"):
            transformer_metrics_file = os.path.join(transformer_results_dir, file)
            break
    
    if not transformer_metrics_file:
        print("No Transformer metrics found.")
        return
        
    with open(transformer_metrics_file, 'r') as f:
        transformer_metrics = json.load(f)
    
    # Create comparison plots
    plt.figure(figsize=(20, 15))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(dqn_metrics['rewards'], alpha=0.3, label='DQN Rewards')
    plt.plot(transformer_metrics['rewards'], alpha=0.3, label='Transformer Rewards')
    
    # Calculate and plot moving averages
    window_size = 20
    if len(dqn_metrics['rewards']) > window_size:
        dqn_moving_avg = np.convolve(dqn_metrics['rewards'], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
        x_avg = np.arange(window_size-1, len(dqn_metrics['rewards']))
        plt.plot(x_avg, dqn_moving_avg, 'r-', label='DQN Moving Avg', linewidth=2)
        
    if len(transformer_metrics['rewards']) > window_size:
        transformer_moving_avg = np.convolve(transformer_metrics['rewards'], 
                                          np.ones(window_size)/window_size, 
                                          mode='valid')
        x_avg = np.arange(window_size-1, len(transformer_metrics['rewards']))
        plt.plot(x_avg, transformer_moving_avg, 'b-', label='Transformer Moving Avg', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Reward Comparison')
    plt.grid(True)
    
    # Plot task completion rates
    plt.subplot(2, 2, 2)
    if 'task_completion_rates' in dqn_metrics and 'task_completion_rates' in transformer_metrics:
        plt.plot(dqn_metrics['task_completion_rates'], alpha=0.3, label='DQN')
        plt.plot(transformer_metrics['task_completion_rates'], alpha=0.3, label='Transformer')
        
        # Calculate and plot moving averages
        if len(dqn_metrics['task_completion_rates']) > window_size:
            dqn_moving_avg = np.convolve(dqn_metrics['task_completion_rates'], 
                                       np.ones(window_size)/window_size, 
                                       mode='valid')
            x_avg = np.arange(window_size-1, len(dqn_metrics['task_completion_rates']))
            plt.plot(x_avg, dqn_moving_avg, 'r-', label='DQN Moving Avg', linewidth=2)
            
        if len(transformer_metrics['task_completion_rates']) > window_size:
            transformer_moving_avg = np.convolve(transformer_metrics['task_completion_rates'], 
                                              np.ones(window_size)/window_size, 
                                              mode='valid')
            x_avg = np.arange(window_size-1, len(transformer_metrics['task_completion_rates']))
            plt.plot(x_avg, transformer_moving_avg, 'b-', label='Transformer Moving Avg', linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Completion Rate')
        plt.legend()
        plt.title('Task Completion Rate Comparison')
        plt.grid(True)
    
    # Plot latencies
    plt.subplot(2, 2, 3)
    if 'latencies' in dqn_metrics and 'latencies' in transformer_metrics:
        plt.plot(dqn_metrics['latencies'], alpha=0.3, label='DQN')
        plt.plot(transformer_metrics['latencies'], alpha=0.3, label='Transformer')
        
        # Calculate and plot moving averages
        if len(dqn_metrics['latencies']) > window_size:
            dqn_moving_avg = np.convolve(dqn_metrics['latencies'], 
                                       np.ones(window_size)/window_size, 
                                       mode='valid')
            x_avg = np.arange(window_size-1, len(dqn_metrics['latencies']))
            plt.plot(x_avg, dqn_moving_avg, 'r-', label='DQN Moving Avg', linewidth=2)
            
        if len(transformer_metrics['latencies']) > window_size:
            transformer_moving_avg = np.convolve(transformer_metrics['latencies'], 
                                              np.ones(window_size)/window_size, 
                                              mode='valid')
            x_avg = np.arange(window_size-1, len(transformer_metrics['latencies']))
            plt.plot(x_avg, transformer_moving_avg, 'b-', label='Transformer Moving Avg', linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Latency (s)')
        plt.legend()
        plt.title('Latency Comparison')
        plt.grid(True)
    
    # Plot node usage
    plt.subplot(2, 2, 4)
    if 'node_usage' in dqn_metrics and 'node_usage' in transformer_metrics:
        plt.plot(dqn_metrics['node_usage'], alpha=0.3, label='DQN')
        plt.plot(transformer_metrics['node_usage'], alpha=0.3, label='Transformer')
        
        # Calculate and plot moving averages
        if len(dqn_metrics['node_usage']) > window_size:
            dqn_moving_avg = np.convolve(dqn_metrics['node_usage'], 
                                       np.ones(window_size)/window_size, 
                                       mode='valid')
            x_avg = np.arange(window_size-1, len(dqn_metrics['node_usage']))
            plt.plot(x_avg, dqn_moving_avg, 'r-', label='DQN Moving Avg', linewidth=2)
            
        if len(transformer_metrics['node_usage']) > window_size:
            transformer_moving_avg = np.convolve(transformer_metrics['node_usage'], 
                                              np.ones(window_size)/window_size, 
                                              mode='valid')
            x_avg = np.arange(window_size-1, len(transformer_metrics['node_usage']))
            plt.plot(x_avg, transformer_moving_avg, 'b-', label='Transformer Moving Avg', linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Node Usage')
        plt.legend()
        plt.title('Node Usage Comparison')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_filename = os.path.join(save_dir, f"model_comparison_{timestamp}.png")
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Table comparison of final performance
    final_window = min(100, min(len(dqn_metrics['rewards']), len(transformer_metrics['rewards'])))
    
    comparison_table = {
        'Model': ['DQN', 'Transformer'],
        'Avg Reward': [
            np.mean(dqn_metrics['rewards'][-final_window:]),
            np.mean(transformer_metrics['rewards'][-final_window:])
        ]
    }
    
    # Add task completion rate if available
    if 'task_completion_rates' in dqn_metrics and 'task_completion_rates' in transformer_metrics:
        comparison_table['Avg Completion Rate'] = [
            np.mean(dqn_metrics['task_completion_rates'][-final_window:]),
            np.mean(transformer_metrics['task_completion_rates'][-final_window:])
        ]
    
    # Add latency if available
    if 'latencies' in dqn_metrics and 'latencies' in transformer_metrics:
        comparison_table['Avg Latency'] = [
            np.mean(dqn_metrics['latencies'][-final_window:]),
            np.mean(transformer_metrics['latencies'][-final_window:])
        ]
    
    # Add node usage if available
    if 'node_usage' in dqn_metrics and 'node_usage' in transformer_metrics:
        comparison_table['Avg Node Usage'] = [
            np.mean(dqn_metrics['node_usage'][-final_window:]),
            np.mean(transformer_metrics['node_usage'][-final_window:])
        ]
    
    # Save comparison table to JSON
    table_filename = os.path.join(save_dir, f"model_comparison_table_{timestamp}.json")
    with open(table_filename, 'w') as f:
        json.dump(comparison_table, f, indent=4)
    
    # Print comparison table
    print("\nPerformance Comparison (last 100 episodes):")
    print(f"{'Model':<12} {'Avg Reward':<15} {'Completion Rate':<20} {'Latency':<15} {'Node Usage':<15}")
    print("-" * 80)
    for i in range(2):
        row = f"{comparison_table['Model'][i]:<12} {comparison_table['Avg Reward'][i]:<15.4f} "
        if 'Avg Completion Rate' in comparison_table:
            row += f"{comparison_table['Avg Completion Rate'][i]:<20.4f} "
        else:
            row += f"{'N/A':<20} "
        
        if 'Avg Latency' in comparison_table:
            row += f"{comparison_table['Avg Latency'][i]:<15.4f} "
        else:
            row += f"{'N/A':<15} "
            
        if 'Avg Node Usage' in comparison_table:
            row += f"{comparison_table['Avg Node Usage'][i]:<15.4f}"
        else:
            row += f"{'N/A':<15}"
            
        print(row)
    
    print(f"\nComparison results saved to {comparison_filename} and {table_filename}")

if __name__ == "__main__":
    # Set timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configuration
    env_config = {
        'sumo_config': 'astana.sumocfg',
        'simulation_duration': 300,
        'time_step': 1,
        'queue_process_interval': 5,
        'max_queue_length': 50,
        'history_length': 10,
        'seed': 42,
        'num_episodes': 1000,
        'max_steps': 300
    }
    
    # Create directory for this specific run
    run_dir = os.path.join("transformer_results", f"run_{timestamp}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # Train the agent
    print(f"Starting transformer training with configuration: {env_config}")
    agent, metrics = train_vec_transformer(env_config)
    
    # Save results
    save_transformer_training_results(metrics, env_config, save_dir=run_dir)
    
    print(f"Training completed. Results saved to {run_dir}")
    
    # Optional: Compare with DQN model if available
    dqn_results_dir = "results"  # Directory containing DQN results
    
    if os.path.exists(dqn_results_dir):
        print("Comparing Transformer with DQN results...")
        compare_models(dqn_results_dir, run_dir)
    else:
        print("No DQN results directory found for comparison.")