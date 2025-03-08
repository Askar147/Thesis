import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data, Batch
import random
import matplotlib.pyplot as plt
import os
from collections import deque
from datetime import datetime
import json

# Import the MEC environment
from mec_environment import MECEnvironment

class EdgeConv(MessagePassing):
    """Enhanced edge convolution layer for MEC with improved message passing"""
    def __init__(self, in_channels, out_channels, aggr='max'):
        super(EdgeConv, self).__init__(aggr=aggr)
        
        # More expressive MLP for edge features
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(0.2)
        )
        
        # Edge attention mechanism
        self.edge_attention = nn.Sequential(
            nn.Linear(in_channels * 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr=None):
        # Concatenate source and target node features
        tmp = torch.cat([x_i, x_j], dim=1)
        
        # Apply attention if available
        if edge_attr is not None:
            # Compute attention weight
            attention = self.edge_attention(tmp)
            return self.mlp(tmp) * attention
        
        return self.mlp(tmp)

class MECGraphNet(nn.Module):
    """Graph Neural Network for MEC task offloading with multi-head attention and skip connections"""
    def __init__(self, node_features, edge_features, num_servers, 
                 hidden_dim=128, num_layers=4, heads=4,
                 dropout=0.1, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(MECGraphNet, self).__init__()
        
        # Device setup
        self.device = device
        
        # Dimensions
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_servers = num_servers
        self.hidden_dim = hidden_dim
        self.heads = heads
        
        # Node feature processing with better normalization
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        ).to(device)
        
        # Edge feature processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        ).to(device)
        
        # Graph convolution layers with different aggregations
        self.conv_layers = nn.ModuleList([
            EdgeConv(hidden_dim, hidden_dim, aggr='max' if i % 2 == 0 else 'mean').to(device) 
            for i in range(num_layers)
        ])
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout).to(device) 
            for _ in range(num_layers)
        ])
        
        # Layer normalization after each block
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim).to(device) for _ in range(num_layers)
        ])
        
        # Dueling network architecture
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_servers + 1)  # +1 for local processing
        ).to(device)
        
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def create_graph_from_observation(self, obs):
        """Convert MEC environment observation to graph structure"""
        # Extract state components
        task_size = obs['task_size']
        required_cpu_cycles = obs['required_cpu_cycles']
        task_deadline = obs['task_deadline']
        mv_compute_capacity = obs['mv_compute_capacity']
        mv_local_queue = obs['mv_local_queue']
        mv_energy = obs['mv_energy']
        es_compute_capacity = obs['es_compute_capacity']
        es_queue_length = obs['es_queue_length']
        channel_gain = obs['channel_gain']
        network_delay = obs['network_delay']
        
        # Focus on a single mobile vehicle (MV) - the first one
        mv_idx = 0
        
        # Convert numpy arrays to tensors
        task_size_tensor = torch.FloatTensor(task_size[mv_idx]).to(self.device)
        required_cpu_cycles_tensor = torch.FloatTensor(required_cpu_cycles[mv_idx]).to(self.device)
        task_deadline_tensor = torch.FloatTensor(task_deadline[mv_idx]).to(self.device)
        mv_compute_tensor = torch.FloatTensor(mv_compute_capacity[mv_idx]).to(self.device)
        mv_queue_tensor = torch.FloatTensor(mv_local_queue[mv_idx]).to(self.device)
        mv_energy_tensor = torch.FloatTensor(mv_energy[mv_idx]).to(self.device)
        
        # Create MV node feature vector
        mv_features = torch.cat([
            task_size_tensor,
            required_cpu_cycles_tensor,
            task_deadline_tensor,
            mv_compute_tensor,
            mv_queue_tensor,
            mv_energy_tensor
        ]).view(1, -1)  # Shape: [1, 6]
        
        # Create edge server (ES) node feature vectors
        es_features_list = []
        for es_idx in range(self.num_servers):
            es_compute = torch.FloatTensor(es_compute_capacity[es_idx]).to(self.device)
            es_queue = torch.FloatTensor(es_queue_length[es_idx]).to(self.device)
            
            # Channel gain and network delay between this MV and this ES
            es_channel_gain = torch.FloatTensor([channel_gain[mv_idx][es_idx]]).to(self.device)
            es_network_delay = torch.FloatTensor([network_delay[mv_idx][es_idx]]).to(self.device)
            
            # Combined ES features
            es_feature_vec = torch.cat([
                es_compute,
                es_queue,
                es_channel_gain,
                es_network_delay
            ]).view(1, -1)  # Shape: [1, 4]
            
            es_features_list.append(es_feature_vec)
        
        # Stack all ES features
        es_features = torch.cat(es_features_list, dim=0)  # Shape: [num_servers, 4]
        
        # Pad ES features to match MV feature dimension
        padding = torch.zeros(es_features.size(0), mv_features.size(1) - es_features.size(1)).to(self.device)
        es_features_padded = torch.cat([es_features, padding], dim=1)  # Shape: [num_servers, 6]
        
        # Combine MV and ES features
        all_node_features = torch.cat([mv_features, es_features_padded], dim=0)  # Shape: [num_servers + 1, 6]
        
        # Create edges
        edge_index = []
        edge_attr = []
        
        # Add edges from MV to each ES
        for es_idx in range(self.num_servers):
            # Features for the edge between MV and this ES
            edge_features = [
                float(channel_gain[mv_idx][es_idx]),         # Channel gain
                float(network_delay[mv_idx][es_idx]),        # Network delay
                float(es_compute_capacity[es_idx][0]),       # ES compute capacity
                float(es_queue_length[es_idx][0]),           # ES queue length
                float(task_size[mv_idx][0]),                 # Task size
                float(required_cpu_cycles[mv_idx][0])        # Required CPU cycles
            ]
            
            # MV -> ES
            edge_index.append([0, es_idx + 1])  # MV is node 0, ES starts from 1
            edge_attr.append(edge_features)
            
            # ES -> MV (for bidirectional information flow)
            edge_index.append([es_idx + 1, 0])
            edge_attr.append(edge_features)
        
        # Add ES to ES edges for server cooperation modeling
        for i in range(self.num_servers):
            for j in range(i+1, self.num_servers):
                # Create server-to-server edge features
                es_to_es_features = [
                    float(es_compute_capacity[i][0]),       # Source ES compute capacity
                    float(es_compute_capacity[j][0]),       # Target ES compute capacity
                    float(es_queue_length[i][0]),           # Source ES queue length
                    float(es_queue_length[j][0]),           # Target ES queue length
                    0.0,                                     # Placeholder
                    0.0                                      # Placeholder
                ]
                
                # ES i -> ES j
                edge_index.append([i + 1, j + 1])
                edge_attr.append(es_to_es_features)
                
                # ES j -> ES i
                edge_index.append([j + 1, i + 1])
                edge_attr.append(es_to_es_features)
        
        # Convert to PyTorch tensors
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device=self.device).t()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float, device=self.device)
        
        # Calculate edge feature dimension
        edge_feat_dim = edge_attr_tensor.size(1)
        
        # Ensure edge features have the right dimension
        if edge_feat_dim != self.edge_features:
            # If dimensions don't match, pad or truncate
            if edge_feat_dim < self.edge_features:
                padding = torch.zeros(edge_attr_tensor.size(0), self.edge_features - edge_feat_dim, device=self.device)
                edge_attr_tensor = torch.cat([edge_attr_tensor, padding], dim=1)
            else:
                edge_attr_tensor = edge_attr_tensor[:, :self.edge_features]
        
        # Create PyG Data object
        data = Data(
            x=all_node_features,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor
        )
        
        return data
    
    def forward(self, obs):
        # Create graph from observation
        data = self.create_graph_from_observation(obs)
        
        # Process node features
        x = self.node_encoder(data.x)
        
        # Process edge features if present
        if data.edge_attr is not None:
            edge_attr = self.edge_encoder(data.edge_attr)
        else:
            edge_attr = None
        
        # Apply graph convolutions with residual connections and layer normalization
        x_residual = x
        for i, (conv, attention, norm) in enumerate(zip(self.conv_layers, self.attention_layers, self.layer_norms)):
            # Apply convolution
            x_conv = conv(x, data.edge_index, edge_attr)
            
            # Apply attention
            x_attention = attention(x, data.edge_index)
            
            # Combine with residual connection
            x = x_residual + x_conv + x_attention
            
            # Apply normalization and non-linearity
            x = norm(x)
            x = F.leaky_relu(x, 0.2)
            
            # Update residual
            x_residual = x
        
        # Get MV node representation (first node)
        mv_repr = x[0]
        
        # Dueling architecture: split into advantage and value
        advantage = self.advantage(mv_repr)
        value = self.value(mv_repr)
        
        # Combine value and advantage for Q-values
        q_values = value + (advantage - advantage.mean())
        
        return q_values, value

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for more efficient learning"""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-4):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.alpha = alpha  # Controls how much prioritization is used
        self.beta = beta    # Controls importance sampling weights
        self.beta_increment = beta_increment  # Beta annealing
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        # Store with max priority for new experiences
        max_priority = self.max_priority if len(self.buffer) > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        N = len(self.buffer)
        if N == 0:
            return [], [], []
            
        # Calculate sampling probabilities based on priority
        priorities = self.priorities[:N] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(N, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Increment beta for annealing
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Return sampled experiences, weights, and indices
        samples = [self.buffer[idx] for idx in indices]
        weights = torch.tensor(weights, dtype=torch.float32)
        return samples, weights, indices
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class MECGNNAgent:
    """GNN-based agent for MEC task offloading with PER and dueling architecture"""
    def __init__(self, 
                num_edge_servers,
                node_features=6,  # Updated for MEC environment
                edge_features=6,  # Updated for MEC environment
                hidden_dim=128, 
                num_heads=4, 
                num_layers=4,
                lr=0.0001, 
                batch_size=256,
                min_replay_size=10000,
                device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.num_edge_servers = num_edge_servers
        self.action_size = num_edge_servers + 1  # Local + edge servers
        
        print(f"Using device: {self.device}")
        print(f"Number of edge servers: {num_edge_servers}")
        print(f"Action size: {self.action_size}")
        
        # Initialize networks
        self.policy_net = MECGraphNet(
            node_features=node_features,
            edge_features=edge_features,
            num_servers=num_edge_servers,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=num_heads,
            device=device
        )
        
        self.target_net = MECGraphNet(
            node_features=node_features,
            edge_features=edge_features,
            num_servers=num_edge_servers,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=num_heads,
            device=device
        )
        
        # Copy weights from policy to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Freeze target network parameters
        for param in self.target_net.parameters():
            param.requires_grad = False
        
        # Initialize optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), 
            lr=lr,
            weight_decay=1e-5,
            amsgrad=True
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=10,
            verbose=True
        )
        
        # Initialize prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(100000)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.01  # Soft update parameter
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        
        self.epsilon = 1.0  # Starting exploration rate
        self.epsilon_min = 0.05  # Minimum exploration rate
        self.epsilon_decay_steps = 2500  # Steps for epsilon decay
        self.current_step = 0
        
        self.update_frequency = 4  # Update target network every 4 steps
        
        # Multi-step returns
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step)
        
        # Tracking metrics
        self.recent_losses = deque(maxlen=100)
    
    def _get_n_step_info(self):
        """Calculate n-step returns for temporal difference learning"""
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
            
        return reward, next_state, done
    
    def select_action(self, state, evaluate=False):
        """Select action using epsilon-greedy policy"""
        # Update epsilon with linear decay
        if self.current_step < self.epsilon_decay_steps:
            self.epsilon = max(self.epsilon_min, 1.0 - (self.current_step / self.epsilon_decay_steps) * (1.0 - self.epsilon_min))
        else:
            self.epsilon = self.epsilon_min
            
        # Random exploration
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Greedy action from policy network
        with torch.no_grad():
            q_values, _ = self.policy_net(state)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition with n-step returns"""
        # Store in n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If n-step buffer is full, process and add to replay buffer
        if len(self.n_step_buffer) == self.n_step:
            # Get first state and action
            state, action = self.n_step_buffer[0][:2]
            
            # Get reward, next_state, and done flag from n-step return
            reward, next_state, done = self._get_n_step_info()
            
            # Store in replay buffer
            self.replay_buffer.push(state, action, reward, next_state, done)
        
        # If episode ends before n-step buffer is full
        if done:
            while len(self.n_step_buffer) > 0:
                # Get the first state and action
                state, action = self.n_step_buffer[0][:2]
                
                # Calculate n-step returns
                reward, next_state, done = self._get_n_step_info()
                
                # Store in replay buffer
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Remove the first entry
                self.n_step_buffer.popleft()
    
    def train(self):
        """Train the agent using prioritized experience replay"""
        # Only train if we have enough samples
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        self.current_step += 1
        
        # Only update every few steps for stability
        if self.current_step % self.update_frequency != 0:
            return None
        
        # Sample from replay buffer with priorities
        transitions, weights, indices = self.replay_buffer.sample(self.batch_size)
        if not transitions:  # Empty buffer case
            return None
            
        batch = list(zip(*transitions))
        
        # Process batch
        state_batch = batch[0]  # List of observation dictionaries
        action_batch = torch.tensor(batch[1], device=self.device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=self.device)
        next_state_batch = batch[3]  # List of observation dictionaries
        done_batch = torch.tensor(batch[4], dtype=torch.float32, device=self.device)
        
        # Import weights to device
        weights = weights.to(self.device)
        
        # Get current Q values
        current_q_values = []
        for state in state_batch:
            q_values, _ = self.policy_net(state)
            current_q_values.append(q_values)
        current_q_values = torch.stack(current_q_values)
        current_q = current_q_values.gather(1, action_batch.unsqueeze(1))
        
        # Get target Q values using Double DQN approach
        target_q = []
        with torch.no_grad():
            target_q_list = []
            for next_state in next_state_batch:
                # Get actions from policy network
                next_q_values, _ = self.policy_net(next_state)
                best_action = next_q_values.argmax().item()  # Take scalar value
                
                # Get Q-values from target network
                target_q_values, _ = self.target_net(next_state)
                
                # Use DDQN: select action with policy network, evaluate with target network
                next_q = target_q_values[best_action].unsqueeze(0)  # Make sure it's a tensor
                target_q_list.append(next_q)
            
            target_q = torch.stack(target_q_list).to(self.device)
            target_q = reward_batch.unsqueeze(1) + \
                    (1.0 - done_batch.unsqueeze(1)) * self.gamma * target_q
        # Compute loss with priorities
        errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        # Update replay buffer priorities
        priority_updates = [float(error) + 1e-6 for error in errors.flatten()]
        self.replay_buffer.update_priorities(indices, priority_updates)
        
        # Soft update target network
        if self.current_step % self.update_frequency == 0:
            for target_param, policy_param in zip(self.target_net.parameters(),
                                                self.policy_net.parameters()):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1 - self.tau) * target_param.data
                )
        
        # Store loss for monitoring
        self.recent_losses.append(loss.item())
        
        return loss.item()
    
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

def train_mec_gnn(env_config):
    """Training loop for MEC environment with GNN"""
    # Environment setup
    env = MECEnvironment(num_mvs=env_config['num_mvs'], 
                         num_edge_servers=env_config['num_edge_servers'],
                         continuous_action=False,
                         difficulty=env_config['difficulty'])
    
    # Initialize GNN agent
    agent = MECGNNAgent(
        num_edge_servers=env.num_edge_servers,
        node_features=6,
        edge_features=6,
        hidden_dim=128,
        num_heads=4,
        num_layers=4,
        lr=0.0003,
        batch_size=128,
        min_replay_size=5000
    )
    
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
        'processing_times': [],
        'energy_consumption': []
    }
    
    # Pre-fill replay buffer with random experiences
    print("Pre-filling replay buffer with random experiences...")
    obs = env.reset()
    
    # Fill buffer until we reach min_replay_size
    while len(agent.replay_buffer) < agent.min_replay_size:
        action = random.randrange(agent.action_size)
        next_obs, rewards, done, info = env.step(action)
        reward = rewards[0]  # First MV
        
        # Store transition
        agent.store_transition(obs, action, reward, next_obs, done)
        
        obs = next_obs
        if done:
            obs = env.reset()
            
        # Print progress occasionally
        if len(agent.replay_buffer) % 1000 == 0:
            print(f"Replay buffer: {len(agent.replay_buffer)}/{agent.min_replay_size}")
    
    print("Replay buffer filled. Starting training...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        
        episode_reward = 0
        episode_losses = []
        episode_completion_rates = []
        episode_processing_times = []
        episode_energy = []
        
        # Clear n-step buffer at start of episode
        agent.n_step_buffer.clear()
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(obs)
            next_obs, rewards, done, info = env.step(action)
            
            # Get reward for the specific MV we're training
            reward = rewards[0]  # First MV in our case
            
            # Store transition and train
            agent.store_transition(obs, action, reward, next_obs, done)
            loss = agent.train()
            
            if loss is not None:
                episode_losses.append(loss)
            
            # Track metrics
            episode_completion_rates.append(info['task_completion_rate'])
            episode_processing_times.append(info['avg_processing_time'])
            episode_energy.append(info['avg_energy_consumption'])
            
            obs = next_obs
            episode_reward += reward
            
            if done:
                break
        
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
            
            # Update learning rate scheduler
            agent.scheduler.step(avg_reward)
        
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
            
            # Save model periodically
            if episode > 0 and episode % (eval_frequency * 5) == 0:
                agent.save_model(f"gnn_model_episode_{episode}.pt")
                
    # Save final model
    agent.save_model("gnn_model_final.pt")
    
    return agent, metrics

def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate agent performance without exploration"""
    total_rewards = []
    task_completion_rates = []
    processing_times = []
    energy_consumption = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_completion_rates = []
        episode_processing_times = []
        episode_energy = []
        
        max_steps = 100
        for step in range(max_steps):
            # Select action deterministically
            action = agent.select_action(obs, evaluate=True)
            
            # Take action
            next_obs, rewards, done, info = env.step(action)
            
            # Track metrics
            episode_completion_rates.append(info['task_completion_rate'])
            episode_processing_times.append(info['avg_processing_time'])
            episode_energy.append(info['avg_energy_consumption'])
            
            # Update state and reward
            episode_reward += rewards[0]  # First MV
            obs = next_obs
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        task_completion_rates.append(np.mean(episode_completion_rates))
        processing_times.append(np.mean(episode_processing_times))
        energy_consumption.append(np.mean(episode_energy))
    
    # Return average metrics
    return {
        'avg_reward': np.mean(total_rewards),
        'avg_completion_rate': np.mean(task_completion_rates),
        'avg_processing_time': np.mean(processing_times),
        'avg_energy_consumption': np.mean(energy_consumption)
    }

def save_training_results(metrics, env_config, save_dir="gnn_results"):
    """Save training metrics and plots to specified directory"""
    # Create results directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save configuration
    config_filename = os.path.join(save_dir, f"gnn_config_{timestamp}.json")
    with open(config_filename, 'w') as f:
        json.dump(env_config, f, indent=4)
    
    # Save metrics to JSON file
    metrics_filename = os.path.join(save_dir, f"gnn_metrics_{timestamp}.json")
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
    plot_filename = os.path.join(save_dir, f"gnn_training_plot_{timestamp}.png")
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
    plot2_filename = os.path.join(save_dir, f"gnn_training_plot2_{timestamp}.png")
    plt.savefig(plot2_filename, dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print(f"Results saved to directory: {save_dir}")
    print(f"Config saved as: gnn_config_{timestamp}.json")
    print(f"Metrics saved as: gnn_metrics_{timestamp}.json")
    print(f"Plots saved as: gnn_training_plot_{timestamp}.png and gnn_training_plot2_{timestamp}.png")

def compare_models(ddqn_results_dir, gnn_results_dir, save_dir="comparison_results"):
    """Compare DDQN and GNN model performances"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load DDQN metrics
    ddqn_metrics_file = None
    for file in os.listdir(ddqn_results_dir):
        if file.startswith("metrics_") and file.endswith(".json"):
            ddqn_metrics_file = os.path.join(ddqn_results_dir, file)
            break
    
    if not ddqn_metrics_file:
        print("No DDQN metrics found.")
        return
    
    with open(ddqn_metrics_file, 'r') as f:
        ddqn_metrics = json.load(f)
    
    # Load GNN metrics
    gnn_metrics_file = None
    for file in os.listdir(gnn_results_dir):
        if file.startswith("gnn_metrics_") and file.endswith(".json"):
            gnn_metrics_file = os.path.join(gnn_results_dir, file)
            break
    
    if not gnn_metrics_file:
        print("No GNN metrics found.")
        return
        
    with open(gnn_metrics_file, 'r') as f:
        gnn_metrics = json.load(f)
    
    # Create comparison plots
    plt.figure(figsize=(20, 15))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(ddqn_metrics['rewards'], alpha=0.5, label='DDQN Rewards')
    plt.plot(gnn_metrics['rewards'], alpha=0.5, label='GNN Rewards')
    
    # Calculate and plot moving averages
    window_size = 20
    if len(ddqn_metrics['rewards']) > window_size:
        ddqn_moving_avg = np.convolve(ddqn_metrics['rewards'], 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
        x_avg = np.arange(window_size-1, len(ddqn_metrics['rewards']))
        plt.plot(x_avg, ddqn_moving_avg, 'r-', label='DDQN Moving Avg', linewidth=2)
        
    if len(gnn_metrics['rewards']) > window_size:
        gnn_moving_avg = np.convolve(gnn_metrics['rewards'], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
        x_avg = np.arange(window_size-1, len(gnn_metrics['rewards']))
        plt.plot(x_avg, gnn_moving_avg, 'g-', label='GNN Moving Avg', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Reward Comparison')
    plt.grid(True)
    
    # Plot task completion rates
    plt.subplot(2, 2, 2)
    plt.plot(ddqn_metrics['task_completion_rates'], alpha=0.5, label='DDQN')
    plt.plot(gnn_metrics['task_completion_rates'], alpha=0.5, label='GNN')
    
    # Calculate and plot moving averages
    if len(ddqn_metrics['task_completion_rates']) > window_size:
        ddqn_moving_avg = np.convolve(ddqn_metrics['task_completion_rates'], 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
        x_avg = np.arange(window_size-1, len(ddqn_metrics['task_completion_rates']))
        plt.plot(x_avg, ddqn_moving_avg, 'r-', label='DDQN Moving Avg', linewidth=2)
        
    if len(gnn_metrics['task_completion_rates']) > window_size:
        gnn_moving_avg = np.convolve(gnn_metrics['task_completion_rates'], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
        x_avg = np.arange(window_size-1, len(gnn_metrics['task_completion_rates']))
        plt.plot(x_avg, gnn_moving_avg, 'g-', label='GNN Moving Avg', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Completion Rate')
    plt.legend()
    plt.title('Task Completion Rate Comparison')
    plt.grid(True)
    
    # Plot processing times
    plt.subplot(2, 2, 3)
    plt.plot(ddqn_metrics['processing_times'], alpha=0.5, label='DDQN')
    plt.plot(gnn_metrics['processing_times'], alpha=0.5, label='GNN')
    
    # Calculate and plot moving averages
    if len(ddqn_metrics['processing_times']) > window_size:
        ddqn_moving_avg = np.convolve(ddqn_metrics['processing_times'], 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
        x_avg = np.arange(window_size-1, len(ddqn_metrics['processing_times']))
        plt.plot(x_avg, ddqn_moving_avg, 'r-', label='DDQN Moving Avg', linewidth=2)
        
    if len(gnn_metrics['processing_times']) > window_size:
        gnn_moving_avg = np.convolve(gnn_metrics['processing_times'], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
        x_avg = np.arange(window_size-1, len(gnn_metrics['processing_times']))
        plt.plot(x_avg, gnn_moving_avg, 'g-', label='GNN Moving Avg', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Processing Time')
    plt.legend()
    plt.title('Processing Time Comparison')
    plt.grid(True)
    
    # Plot energy consumption
    plt.subplot(2, 2, 4)
    plt.plot(ddqn_metrics['energy_consumption'], alpha=0.5, label='DDQN')
    plt.plot(gnn_metrics['energy_consumption'], alpha=0.5, label='GNN')
    
    # Calculate and plot moving averages
    if len(ddqn_metrics['energy_consumption']) > window_size:
        ddqn_moving_avg = np.convolve(ddqn_metrics['energy_consumption'], 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
        x_avg = np.arange(window_size-1, len(ddqn_metrics['energy_consumption']))
        plt.plot(x_avg, ddqn_moving_avg, 'r-', label='DDQN Moving Avg', linewidth=2)
        
    if len(gnn_metrics['energy_consumption']) > window_size:
        gnn_moving_avg = np.convolve(gnn_metrics['energy_consumption'], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
        x_avg = np.arange(window_size-1, len(gnn_metrics['energy_consumption']))
        plt.plot(x_avg, gnn_moving_avg, 'g-', label='GNN Moving Avg', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Energy Consumption')
    plt.legend()
    plt.title('Energy Consumption Comparison')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_filename = os.path.join(save_dir, f"model_comparison_{timestamp}.png")
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison results saved to {comparison_filename}")

if __name__ == "__main__":
    # Configuration
    env_config = {
        'num_mvs': 5,               # Number of mobile vehicles
        'num_edge_servers': 3,       # Number of edge servers
        'difficulty': 'normal',      # Difficulty level: 'easy', 'normal', or 'hard'
        'num_episodes': 3000,        # Number of training episodes
        'max_steps': 100             # Maximum steps per episode
    }
    
    # Create directory for this specific run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("gnn_results", f"gnn_run_{timestamp}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # Train the agent
    print(f"Starting training with configuration: {env_config}")
    agent, metrics = train_mec_gnn(env_config)
    
    # Save results
    save_training_results(metrics, env_config, save_dir=run_dir)
    
    print(f"Training completed. Results saved to {run_dir}")
    
    # Optional: Compare with DDQN results if available
    ddqn_results_dir = "results"  # Directory containing DDQN results
    if os.path.exists(ddqn_results_dir):
        print("Comparing GNN results with DDQN results...")
        compare_models(ddqn_results_dir, run_dir)
    else:
        print("No DDQN results directory found for comparison.")