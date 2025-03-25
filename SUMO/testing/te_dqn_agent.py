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
    Updated for enhanced environment with link quality and energy efficiency
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
    
    # Add new features from enhanced environment
    if 'link_quality' in obs:
        flattened.append(obs['link_quality'][0])
    if 'energy_efficiency' in obs:
        flattened.append(obs['energy_efficiency'][0])
    
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
        self.rewards = []
        self.epsilons = []
        self.losses = []
        self.avg_rewards = []
        
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
        self.losses.append(loss.item())
        
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