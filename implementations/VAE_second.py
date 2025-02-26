import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
import os
import time
import gym
from gym import spaces

class MECEnvironment(gym.Env):
    """Enhanced MEC Environment with server selection-focused rewards"""
    def __init__(self, num_edge_servers=10, continuous_action=False):
        super().__init__()
        
        self.num_edge_servers = num_edge_servers
        self.continuous_action = continuous_action
        
        # Action space definition
        if continuous_action:
            self.action_space = spaces.Box(
                low=0, high=1, shape=(num_edge_servers,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(num_edge_servers)
        
        # Observation space includes task size, server speeds, loads, network conditions, and distances
        self.observation_space = spaces.Dict({
            'task_size': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'server_speeds': spaces.Box(low=np.zeros(num_edge_servers), high=np.ones(num_edge_servers), dtype=np.float32),
            'server_loads': spaces.Box(low=np.zeros(num_edge_servers), high=np.ones(num_edge_servers), dtype=np.float32),
            'network_conditions': spaces.Box(low=np.zeros(num_edge_servers), high=np.ones(num_edge_servers), dtype=np.float32),
            'server_distances': spaces.Box(low=np.zeros(num_edge_servers), high=np.ones(num_edge_servers), dtype=np.float32)
        })
        
        # Initialize server characteristics with more distinct values to encourage clear preferences
        self.server_speeds = np.random.uniform(0.6, 1.0, num_edge_servers)  # Wider range for more differentiation
        self.server_distances = np.random.uniform(0.1, 0.9, num_edge_servers)  # Wider range
        self.bandwidth_up = np.random.uniform(0.5, 1.0, num_edge_servers)  # Wider range
        self.bandwidth_down = np.random.uniform(0.6, 1.0, num_edge_servers)
        
        # Scaling factors for various latency components
        self.uplink_scale = 0.6
        self.prop_scale = 0.04
        self.downlink_scale = 0.5
        self.queue_factor = 0.9
        
        # Keep track of latency history for normalization
        self.latency_history = deque(maxlen=100)
        self.prev_fluctuation = np.zeros(num_edge_servers)
        
        # Create ranking matrices for servers to aid reward calculation
        self.reset()
    
    def reset(self):
        """Reset environment state with new task and updated loads/conditions"""
        self.current_task_size = np.random.uniform(0.2, 0.8)
        self.server_loads = np.random.uniform(0.1, 0.4, self.num_edge_servers)
        self.network_conditions = np.random.uniform(0.7, 1.0, self.num_edge_servers)
        
        # Calculate initial effective speed (accounting for load)
        self.effective_speeds = self.server_speeds * (1 - 0.8 * self.server_loads)
        
        # Generate rankings of servers by different metrics
        self.speed_ranks = self.get_server_ranks(self.effective_speeds, reverse=True)  # Higher is better
        self.load_ranks = self.get_server_ranks(self.server_loads)  # Lower is better
        self.distance_ranks = self.get_server_ranks(self.server_distances)  # Lower is better
        self.network_ranks = self.get_server_ranks(self.network_conditions, reverse=True)  # Higher is better
        
        return self._get_observation()
    
    def get_server_ranks(self, values, reverse=False):
        """Generate server rankings based on a metric (lower rank is better unless reversed)"""
        # Argsort gives indices that would sort the array
        if reverse:
            # For metrics where higher is better (speeds, network conditions)
            return np.argsort(np.argsort(-values))
        else:
            # For metrics where lower is better (loads, distances)
            return np.argsort(np.argsort(values))
    
    def _calculate_total_latency(self, server_idx):
        """Calculate total latency with all components"""
        # 1. Uplink transmission delay
        uplink_delay = (self.current_task_size / self.bandwidth_up[server_idx]) * \
                       (1 / self.network_conditions[server_idx]) * self.uplink_scale
        
        # 2. Propagation delay based on distance
        prop_delay = self.server_distances[server_idx] * self.prop_scale
        
        # 3. Processing delay (affected by server load)
        effective_speed = self.server_speeds[server_idx] * (1 - 0.8 * self.server_loads[server_idx])
        processing_delay = self.current_task_size / max(effective_speed, 0.1)
        
        # 4. Downlink transmission delay
        result_size = self.current_task_size * 0.05  # Result is smaller than input
        downlink_delay = (result_size / self.bandwidth_down[server_idx]) * \
                         (1 / self.network_conditions[server_idx]) * self.downlink_scale
        
        # 5. Queuing delay (scaled by server load)
        queue_delay = self.server_loads[server_idx] * processing_delay * self.queue_factor
        
        total_delay = uplink_delay + prop_delay + processing_delay + downlink_delay + queue_delay
        return total_delay
    
    def step(self, action):
        """Take an action and return next state, reward, done flag, and info"""
        if self.continuous_action:
            action_probs = F.softmax(torch.FloatTensor(action), dim=0).numpy()
            selected_server = np.argmax(action_probs)
        else:
            selected_server = action
        
        # Calculate latency for the chosen server
        total_latency = self._calculate_total_latency(selected_server)
        self.latency_history.append(total_latency)
        
        # Create reward components with stronger emphasis on server selection
        
        # 1. Latency component (normalized against history)
        if len(self.latency_history) > 10:
            avg_latency = np.mean(self.latency_history)
            std_latency = np.std(self.latency_history) + 0.1  # Avoid division by zero
            latency_score = (total_latency - avg_latency) / std_latency
            latency_reward = -1.0 - np.clip(latency_score, -1.0, 1.0)  # Range: -2.0 to 0.0
        else:
            latency_reward = -total_latency / 2.0
        
        # 2. Server selection component (increased weight)
        # Calculate a composite rank considering multiple factors
        composite_rank = (self.speed_ranks[selected_server] * 0.4 + 
                          self.load_ranks[selected_server] * 0.3 + 
                          self.distance_ranks[selected_server] * 0.15 + 
                          self.network_ranks[selected_server] * 0.15)
        
        normalized_rank = composite_rank / self.num_edge_servers
        
        # Higher reward for selecting better servers (0.0 to 0.8)
        selection_reward = 0.8 * (1.0 - normalized_rank)
        
        # 3. Load balancing component
        load_variance = np.var(self.server_loads)
        load_balance_reward = 0.2 * (1.0 - min(load_variance * 4, 1.0))  # 0.0 to 0.2
        
        # 4. Optimal server bonus
        effective_speeds = self.server_speeds * (1 - 0.8 * self.server_loads)
        optimal_server = np.argmax(effective_speeds)
        
        if selected_server == optimal_server:
            optimal_bonus = 0.4  # Substantial bonus for choosing the optimal server
        else:
            # How close to optimal in terms of effective speed
            optimal_ratio = effective_speeds[selected_server] / effective_speeds[optimal_server]
            optimal_bonus = 0.4 * optimal_ratio
        
        # Final reward combines components with server selection having more weight
        reward = latency_reward + selection_reward + load_balance_reward + optimal_bonus
        
        # Scale reward to reasonable range
        reward = np.clip(reward, -2.0, 1.0)
        
        # Update environment state
        self._update_server_loads(selected_server)
        self._update_network_conditions()
        self.current_task_size = np.random.uniform(0.2, 0.8)
        
        # Recalculate effective speeds and ranks
        self.effective_speeds = self.server_speeds * (1 - 0.8 * self.server_loads)
        self.speed_ranks = self.get_server_ranks(self.effective_speeds, reverse=True)
        self.load_ranks = self.get_server_ranks(self.server_loads)
        self.network_ranks = self.get_server_ranks(self.network_conditions, reverse=True)
        
        observation = self._get_observation()
        info = {
            'selected_server': selected_server,
            'server_load': self.server_loads[selected_server],
            'total_latency': total_latency,
            'latency_reward': latency_reward,
            'selection_reward': selection_reward,
            'load_balance_reward': load_balance_reward,
            'optimal_bonus': optimal_bonus,
            'optimal_server': optimal_server
        }
        
        return observation, reward, False, info
    
    def _update_server_loads(self, selected_server):
        """Update server loads with smoother dynamics"""
        # Increase load for the selected server
        self.server_loads[selected_server] = min(
            self.server_loads[selected_server] + self.current_task_size * 0.07,
            0.95  # Cap at 95% to prevent complete saturation
        )
        
        # Apply smoother decay to other servers
        for i in range(self.num_edge_servers):
            if i != selected_server:
                # Linear decay instead of exponential for more stability
                self.server_loads[i] = max(self.server_loads[i] - 0.01, 0.1)
    
    def _update_network_conditions(self):
        """Update network conditions with smoother, temporally correlated fluctuations"""
        # Use smaller fluctuations
        fluctuation = np.random.uniform(-0.03, 0.03, self.num_edge_servers)
        
        # Make fluctuations temporally correlated
        self.prev_fluctuation = 0.7 * self.prev_fluctuation + 0.3 * fluctuation
        
        # Apply the smoothed fluctuations
        self.network_conditions += self.prev_fluctuation
        
        # Ensure values stay in reasonable range
        self.network_conditions = np.clip(self.network_conditions, 0.6, 1.0)
    
    def _get_observation(self):
        """Construct the observation dictionary"""
        return {
            'task_size': np.array([self.current_task_size], dtype=np.float32),
            'server_speeds': self.server_speeds.astype(np.float32),
            'server_loads': self.server_loads.astype(np.float32),
            'network_conditions': self.network_conditions.astype(np.float32),
            'server_distances': self.server_distances.astype(np.float32)
        }


class MECEncoder(nn.Module):
    """Improved encoder network for MEC VAE with residual connections and better normalization"""
    def __init__(self, state_dim, hidden_dim=256, latent_dim=64):  # Increased latent dim
        super(MECEncoder, self).__init__()
        
        # Layer 1
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)  # Light dropout
        
        # Layer 2 with residual connection
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.1)
        
        # Layer 3 with residual connection
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.dropout3 = nn.Dropout(0.1)
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # Layer 1
        x1 = F.elu(self.ln1(self.fc1(x)))  # ELU activation
        x1 = self.dropout1(x1)
        
        # Layer 2 with residual connection
        x2 = F.elu(self.ln2(self.fc2(x1)))
        x2 = self.dropout2(x2)
        x2 = x2 + x1  # Residual connection
        
        # Layer 3 with residual connection
        x3 = F.elu(self.ln3(self.fc3(x2)))
        x3 = self.dropout3(x3)
        x3 = x3 + x2  # Residual connection
        
        # Mean and log variance with better numerical handling
        mu = self.fc_mu(x3)
        
        # Clamp logvar for numerical stability
        logvar = self.fc_logvar(x3)
        logvar = torch.clamp(logvar, -20, 2)
        
        return mu, logvar


class MECDecoder(nn.Module):
    """Improved decoder network for MEC VAE with residual connections"""
    def __init__(self, latent_dim, state_dim, num_servers, hidden_dim=256):
        super(MECDecoder, self).__init__()
        
        self.num_servers = num_servers
        
        # Shared decoder base
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.dropout3 = nn.Dropout(0.1)
        
        # Separate heads for different components with additional layers
        # Task size head
        self.task_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Server speeds head
        self.server_speeds_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_servers)
        )
        
        # Server loads head
        self.server_loads_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_servers)
        )
        
        # Network conditions head
        self.network_conditions_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_servers)
        )
        
        # Server distances head
        self.server_distances_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_servers)
        )
    
    def forward(self, z):
        # Layer 1
        x1 = F.elu(self.ln1(self.fc1(z)))
        x1 = self.dropout1(x1)
        
        # Layer 2 with residual connection
        x2 = F.elu(self.ln2(self.fc2(x1)))
        x2 = self.dropout2(x2)
        x2 = x2 + x1  # Residual connection
        
        # Layer 3 with residual connection
        x3 = F.elu(self.ln3(self.fc3(x2)))
        x3 = self.dropout3(x3)
        x3 = x3 + x2  # Residual connection
        
        # Generate different components
        task_size = torch.sigmoid(self.task_head(x3))
        server_speeds = torch.sigmoid(self.server_speeds_head(x3))
        server_loads = torch.sigmoid(self.server_loads_head(x3))
        network_conditions = torch.sigmoid(self.network_conditions_head(x3))
        server_distances = torch.sigmoid(self.server_distances_head(x3))
        
        return {
            'task_size': task_size,
            'server_speeds': server_speeds,
            'server_loads': server_loads,
            'network_conditions': network_conditions,
            'server_distances': server_distances
        }


class MECVAE(nn.Module):
    """Improved VAE for MEC system state modeling with better handling of KL divergence"""
    def __init__(self, state_dim, num_servers, hidden_dim=256, latent_dim=64):
        super(MECVAE, self).__init__()
        
        self.encoder = MECEncoder(state_dim, hidden_dim, latent_dim)
        self.decoder = MECDecoder(latent_dim, state_dim, num_servers, hidden_dim)
        
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick with better numerical stability"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            # Clamp std for numerical stability
            std = torch.clamp(std, min=1e-6, max=10.0)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, just return the mean for more stable predictions
            return mu
    
    def forward(self, x):
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoded = self.decoder(z)
        
        return decoded, mu, logvar
    
    def generate(self, num_samples, device):
        """Generate new system states"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            return self.decoder(z)


class PrioritizedReplayBuffer:
    """Enhanced Prioritized Experience Replay buffer with better TD error handling"""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.max_priority = 1.0  # Start with a reasonable priority
    
    def push(self, state, action, reward, next_state, done):
        # Use max priority for new experiences
        priority = self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        # Add a small constant to prevent zero probabilities
        probs = (priorities + 1e-6) ** self.alpha
        probs /= probs.sum()
        
        # Update beta
        self.beta = min(1.0, self.beta + (1.0 - self.beta_start) / self.beta_frames * self.frame)
        self.frame += 1
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Compute importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)
        
        return list(zip(*samples)), indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            # Clamp priorities to prevent extreme values
            priority = np.clip(priority, 1e-6, 50.0)
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class DoubleQNetwork(nn.Module):
    """Enhanced Dueling Double Q-Network with improved initialization"""
    def __init__(self, latent_dim, num_actions, hidden_dim=256):
        super(DoubleQNetwork, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.1)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Initialize weights for better training
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x):
        x = F.elu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.elu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage using dueling architecture
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class MECVAEAgent:
    """VAE-based agent for MEC task offloading with improved training dynamics"""
    def __init__(self, state_dim, num_servers, hidden_dim=256, latent_dim=64,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_dim = state_dim
        self.num_servers = num_servers
        
        # Initialize VAE with larger latent dimension
        self.vae = MECVAE(state_dim, num_servers, hidden_dim, latent_dim).to(device)
        
        # Initialize policy networks
        self.policy_net = DoubleQNetwork(latent_dim, num_servers, hidden_dim).to(device)
        self.target_net = DoubleQNetwork(latent_dim, num_servers, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizers with better learning rates
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=0.0005, weight_decay=1e-5)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003, weight_decay=1e-5)
        
        # Learning rate schedulers with more gradual decay
        self.vae_scheduler = StepLR(self.vae_optimizer, step_size=500, gamma=0.7)  # Slower decay
        self.policy_scheduler = StepLR(self.policy_optimizer, step_size=500, gamma=0.7)
        
        # Initialize replay buffer with higher capacity
        self.replay_buffer = PrioritizedReplayBuffer(150000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.batch_size = 128
        self.min_replay_size = 2000  # Larger minimum replay size
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # Higher minimum exploration
        self.epsilon_decay = 0.998  # Slower decay
        self.vae_train_frequency = 3  # Train VAE more frequently
        self.target_update_frequency = 10
        self.steps = 0
        
        # Beta parameter for VAE KL divergence weight (annealing schedule)
        self.vae_beta_start = 0.01  # Start with very low KL weight
        self.vae_beta_end = 0.1  # Gradually increase to this value
        self.vae_beta = self.vae_beta_start
        self.vae_beta_steps = 100000  # Steps over which to anneal beta
        
        # Gradient clipping value
        self.clip_value = 1.0
        
        # Tracking metrics
        self.training_history = {
            'vae_losses': [],
            'policy_losses': [],
            'reconstruction_errors': [],
            'kl_divergences': [],
            'mean_q_values': []
        }
    
    def state_to_tensor(self, state_dict):
        """Convert state dictionary to tensor"""
        return torch.FloatTensor(np.concatenate([
            state_dict['task_size'],
            state_dict['server_speeds'],
            state_dict['server_loads'],
            state_dict['network_conditions'],
            state_dict['server_distances']
        ])).to(self.device)
    
    def select_action(self, state, evaluate=False):
        """Select action with epsilon-greedy policy and optional exploration noise"""
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.num_servers)
        
        with torch.no_grad():
            # Set models to evaluation mode
            self.vae.eval()
            self.policy_net.eval()
            
            state_tensor = self.state_to_tensor(state).unsqueeze(0)
            mu, _ = self.vae.encoder(state_tensor)
            
            # Add some noise for exploration in latent space (only during training)
            if not evaluate and random.random() < 0.3:
                exploration_noise = torch.randn_like(mu) * 0.1
                mu = mu + exploration_noise
            
            q_values = self.policy_net(mu)
            
            # Set models back to training mode if not evaluating
            if not evaluate:
                self.vae.train()
                self.policy_net.train()
                
            return q_values.max(1)[1].item()
    
    def train_vae(self, states, weights=None):
        """Train VAE with improved KL handling and weighted loss"""
        # Set to training mode
        self.vae.train()
        
        # Forward pass
        decoded, mu, logvar = self.vae(states)
        
        # Split the input states tensor into its components
        num_servers = self.num_servers
        task_target = states[:, :1]
        server_speeds_target = states[:, 1:1+num_servers]
        server_loads_target = states[:, 1+num_servers:1+2*num_servers]
        network_conditions_target = states[:, 1+2*num_servers:1+3*num_servers]
        server_distances_target = states[:, 1+3*num_servers:1+4*num_servers]
        
        # Individual reconstruction losses with different weights
        task_loss = F.mse_loss(decoded['task_size'], task_target, reduction='none').mean(1)
        speeds_loss = F.mse_loss(decoded['server_speeds'], server_speeds_target, reduction='none').mean(1)
        loads_loss = F.mse_loss(decoded['server_loads'], server_loads_target, reduction='none').mean(1)
        network_loss = F.mse_loss(decoded['network_conditions'], network_conditions_target, reduction='none').mean(1)
        distance_loss = F.mse_loss(decoded['server_distances'], server_distances_target, reduction='none').mean(1)
        
        # Weight different components based on their importance to decision-making
        # Loads and speeds are more important for server selection
        recon_loss_per_item = (
            1.0 * task_loss + 
            1.5 * speeds_loss + 
            2.0 * loads_loss + 
            1.0 * network_loss + 
            1.0 * distance_loss
        )
        
        # Calculate KL divergence (per item)
        # Using the closed form KL divergence for Gaussian distributions
        kl_loss_per_item = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # Apply importance weights if provided
        if weights is not None:
            recon_loss = (recon_loss_per_item * weights).mean()
            kl_loss = (kl_loss_per_item * weights).mean()
        else:
            recon_loss = recon_loss_per_item.mean()
            kl_loss = kl_loss_per_item.mean()
        
        # Update beta according to annealing schedule
        self.vae_beta = min(
            self.vae_beta_end,
            self.vae_beta_start + (self.vae_beta_end - self.vae_beta_start) * (self.steps / self.vae_beta_steps)
        )
        
        # Total loss (with adjustable beta coefficient)
        vae_loss = recon_loss + self.vae_beta * kl_loss
        
        # Optimize
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.clip_value)
        self.vae_optimizer.step()
        
        # Track metrics
        self.training_history['vae_losses'].append(vae_loss.item())
        self.training_history['reconstruction_errors'].append(recon_loss.item())
        self.training_history['kl_divergences'].append(kl_loss.item())
        
        return vae_loss.item(), recon_loss.item(), kl_loss.item()
    
    def train(self):
        """Train the agent using prioritized experience replay and double DQN"""
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        # Ensure models are in training mode
        self.vae.train()
        self.policy_net.train()
        
        self.steps += 1
        
        # Sample from replay buffer with priorities
        batch, indices, weights = self.replay_buffer.sample(self.batch_size)
        weights = weights.to(self.device)
        
        # Process states
        state_batch = torch.stack([self.state_to_tensor(s) for s in batch[0]])
        action_batch = torch.tensor(batch[1], dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=self.device)
        next_state_batch = torch.stack([self.state_to_tensor(s) for s in batch[3]])
        done_batch = torch.tensor(batch[4], dtype=torch.float32, device=self.device)
        
        # Train VAE periodically
        if self.steps % self.vae_train_frequency == 0:
            # Train on both current and next states to ensure good latent representations
            combined_states = torch.cat([state_batch, next_state_batch], dim=0)
            combined_weights = torch.cat([weights, weights], dim=0)
            vae_loss, recon_loss, kl_loss = self.train_vae(combined_states, combined_weights)
        else:
            vae_loss, recon_loss, kl_loss = None, None, None
        
        # Get latent representations
        with torch.no_grad():
            # Use encoder without the reparameterization noise during Q-learning
            # This makes training more stable
            state_mu, _ = self.vae.encoder(state_batch)
            next_state_mu, _ = self.vae.encoder(next_state_batch)
        
        # Get current Q values
        current_q = self.policy_net(state_mu)
        current_q_values = current_q.gather(1, action_batch.unsqueeze(1))
        
        # Get target Q values using Double Q-learning
        with torch.no_grad():
            # Get actions from policy network
            next_q_policy = self.policy_net(next_state_mu)
            next_actions = next_q_policy.max(1)[1].unsqueeze(1)
            
            # Get Q-values from target network for those actions
            next_q_target = self.target_net(next_state_mu)
            next_q_values = next_q_target.gather(1, next_actions)
            
            # Compute expected Q values
            expected_q_values = reward_batch.unsqueeze(1) + \
                              (1.0 - done_batch.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute td error for prioritized replay
        td_error = torch.abs(current_q_values - expected_q_values).detach().cpu().numpy()
        
        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_error.squeeze() + 1e-5)
        
        # Compute Huber loss with importance sampling weights
        loss = F.smooth_l1_loss(current_q_values, expected_q_values, reduction='none')
        weighted_loss = (loss * weights.unsqueeze(1)).mean()
        
        # Optimize policy
        self.policy_optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_value)
        self.policy_optimizer.step()
        
        # Track mean Q-value for monitoring
        self.training_history['mean_q_values'].append(current_q.mean().item())
        self.training_history['policy_losses'].append(weighted_loss.item())
        
        # Update target network periodically
        if self.steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Step the learning rate schedulers (less frequently)
        if self.steps % 1000 == 0:
            self.vae_scheduler.step()
            self.policy_scheduler.step()
        
        return weighted_loss.item(), vae_loss, recon_loss, kl_loss
    
    def save_models(self, path):
        """Save both VAE and policy networks"""
        torch.save({
            'vae_state_dict': self.vae.state_dict(),
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'vae_optimizer': self.vae_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'training_history': self.training_history,
            'steps': self.steps
        }, path)
    
    def load_models(self, path):
        """Load both VAE and policy networks"""
        checkpoint = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.training_history = checkpoint['training_history']
        self.steps = checkpoint['steps']


def train_mec_vae():
    """Training loop for improved VAE in MEC environment with better monitoring"""
    # Initialize environment
    env = MECEnvironment(num_edge_servers=10)
    
    # Calculate state size
    state_size = (1 +  # task_size
                env.num_edge_servers +  # server_speeds
                env.num_edge_servers +  # server_loads
                env.num_edge_servers +  # network_conditions
                env.num_edge_servers)   # server_distances
    
    print(f"State size: {state_size}")
    
    # Initialize agent
    agent = MECVAEAgent(state_size, env.num_edge_servers)
    
    # Make sure models start in training mode
    agent.vae.train()
    agent.policy_net.train()
    agent.target_net.eval()
    
    # Training parameters
    num_episodes = 2000
    max_steps = 100
    eval_frequency = 25  # More frequent evaluation
    
    # Early stopping parameters with more patience
    early_stop_patience = 20
    best_reward = -float('inf')
    no_improvement_count = 0
    
    # Metrics tracking
    metrics = {
        'rewards': [],
        'policy_losses': [],
        'vae_losses': [],
        'recon_losses': [],
        'kl_losses': [],
        'latencies': [],
        'server_loads': [],
        'avg_rewards': [],
        'eval_rewards': [],
        'eval_latencies': [],
        'epsilons': []
    }
    
    # Create directory for checkpoints
    os.makedirs('models', exist_ok=True)
    
    # Create evaluation function
    def evaluate_agent(eval_episodes=10):
        eval_rewards = []
        eval_latencies = []
        server_selection_stats = []
        
        # Set models to evaluation mode
        agent.vae.eval()
        agent.policy_net.eval()
        
        for _ in range(eval_episodes):
            state = env.reset()
            episode_reward = 0
            episode_latencies = []
            episode_servers = []
            
            for step in range(max_steps):
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, info = env.step(action)
                
                episode_latencies.append(info['total_latency'])
                episode_servers.append(action)
                state = next_state
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            eval_latencies.append(np.mean(episode_latencies))
            server_selection_stats.append(episode_servers)
        
        # Set models back to training mode
        agent.vae.train()
        agent.policy_net.train()
        
        # Count how many different servers were used
        server_diversity = np.mean([len(set(servers)) for servers in server_selection_stats])
        
        return np.mean(eval_rewards), np.mean(eval_latencies), server_diversity
    
    # Start timer
    start_time = time.time()
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_policy_losses = []
        episode_vae_losses = []
        episode_recon_losses = []
        episode_kl_losses = []
        episode_latencies = []
        episode_servers = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition and train
            agent.replay_buffer.push(state, action, reward, next_state, done)
            losses = agent.train()
            
            if losses is not None:
                policy_loss, vae_loss, recon_loss, kl_loss = losses
                episode_policy_losses.append(policy_loss)
                if vae_loss is not None:
                    episode_vae_losses.append(vae_loss)
                    episode_recon_losses.append(recon_loss)
                    episode_kl_losses.append(kl_loss)
            
            episode_latencies.append(info['total_latency'])
            episode_servers.append(action)
            state = next_state
            episode_reward += reward
        
        # Update epsilon with a better decay schedule
        if episode < 500:
            # Faster decay at the beginning
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        else:
            # Slower decay later to maintain some exploration
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.999)
        
        # Update metrics
        metrics['rewards'].append(episode_reward)
        metrics['latencies'].append(np.mean(episode_latencies))
        metrics['server_loads'].append(np.mean(state['server_loads']))
        metrics['epsilons'].append(agent.epsilon)
        
        if episode_policy_losses:
            metrics['policy_losses'].append(np.mean(episode_policy_losses))
        if episode_vae_losses:
            metrics['vae_losses'].append(np.mean(episode_vae_losses))
            metrics['recon_losses'].append(np.mean(episode_recon_losses))
            metrics['kl_losses'].append(np.mean(episode_kl_losses))
        
        # Calculate running average
        if episode >= eval_frequency:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            metrics['avg_rewards'].append(avg_reward)
        
        # Print progress and evaluate
        if episode % eval_frequency == 0:
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            # Regular evaluation
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            avg_latency = np.mean(metrics['latencies'][-eval_frequency:])
            
            # Run evaluation without exploration
            eval_reward, eval_latency, server_diversity = evaluate_agent()
            metrics['eval_rewards'].append(eval_reward)
            metrics['eval_latencies'].append(eval_latency)
            
            # Calculate server selection diversity
            train_diversity = len(set(episode_servers))
            
            print(f"Episode {episode}/{num_episodes} [{time_str}], "
                  f"Train Reward: {avg_reward:.2f}, "
                  f"Train Latency: {avg_latency:.2f}, "
                  f"Eval Reward: {eval_reward:.2f}, "
                  f"Eval Latency: {eval_latency:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"Server Diversity: {server_diversity:.1f}/{env.num_edge_servers}")
            
            # Early stopping check
            if eval_reward > best_reward:
                best_reward = eval_reward
                no_improvement_count = 0
                # Save the best model
                agent.save_models("models/best_mec_vae.pt")
                print(f"New best model saved with reward: {best_reward:.2f}")
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= early_stop_patience:
                print(f"Early stopping at episode {episode}")
                break
            
            # Save checkpoint every 100 episodes
            if episode % 100 == 0 and episode > 0:
                agent.save_models(f"models/mec_vae_ep{episode}.pt")
    
    # Load the best model for final evaluation
    agent.load_models("models/best_mec_vae.pt")
    
    # Final extensive evaluation
    final_reward, final_latency, final_diversity = evaluate_agent(eval_episodes=50)
    print(f"Final evaluation - Reward: {final_reward:.2f}, Latency: {final_latency:.2f}, Server Diversity: {final_diversity:.1f}/{env.num_edge_servers}")
    
    return agent, metrics


if __name__ == "__main__":
    agent, metrics = train_mec_vae()