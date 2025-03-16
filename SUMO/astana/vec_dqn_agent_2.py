import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime


class DQNNetwork(nn.Module):
    """Deep Q-Network for VEC task offloading decisions"""
    def __init__(self, state_size, action_size, device):
        super(DQNNetwork, self).__init__()
        
        self.device = device
        
        # Print state size for debugging
        print(f"Network initialized with state_size: {state_size}")
        
        # Network layers with better scaling for task offloading decisions
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        
        # Layer normalization for better training stability
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights with orthogonal initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        # Move to device
        self.to(device)
        
    def forward(self, x):
        """Forward pass through the network"""
        x = x.to(self.device)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.ln3(self.fc3(x)))
        return self.fc4(x)


class ReplayBuffer:
    """Experience Replay Buffer for DQN"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for VEC task offloading"""
    def __init__(self, state_size, action_size, bs_id=None):
        self.state_size = state_size
        self.action_size = action_size
        self.bs_id = bs_id  # Base station ID if this agent is for a specific BS
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # DQN hyperparameters
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # starting exploration rate
        self.epsilon_min = 0.05  # minimum exploration rate
        self.epsilon_decay = 0.995  # exploration decay rate
        self.learning_rate = 0.001
        self.batch_size = 64
        self.min_replay_size = 1000
        self.target_update_frequency = 10  # update target network every n episodes
        
        # Create Q-Networks (current and target)
        self.q_network = DQNNetwork(state_size, action_size, self.device)
        self.target_network = DQNNetwork(state_size, action_size, self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training metrics
        self.losses = []
        self.rewards = []
        self.epsilons = []
        self.avg_rewards = []
        self.steps = 0
        self.episodes = 0
    
    def flatten_observation(self, obs):
        """
        Flatten a dictionary observation into a vector suitable for DQN input
        Updated to include task_scenario_id from vec_environment_2
        """
        flattened = []
        
        # Task information
        flattened.append(obs['task_size'][0])
        flattened.append(obs['required_cpu_cycles'][0])
        flattened.append(obs['task_deadline'][0])
        # Include the new scenario ID from vec_environment_2
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
        
        # Historical load information - flatten the history
        flattened.extend(obs['historical_loads'].flatten())
        
        return np.array(flattened, dtype=np.float32)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().argmax().item()
    
    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilons.append(self.epsilon)
    
    def train(self):
        """Train the agent using experience replay"""
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(min(self.batch_size, len(self.replay_buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Compute next Q values with target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        self.steps += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network parameters"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, path):
        """Save model to path"""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model from path"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            self.episodes = checkpoint['episodes']
            print(f"Model loaded from {path}")
            return True
        return False