import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch_geometric.nn import GATConv
from torch.distributions import Normal
import numpy as np
import gym
from gym import spaces
from collections import deque
import random
import os
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class MECEnvironment(gym.Env):
    """Enhanced MEC Environment with server selection-focused rewards"""
    def __init__(self, num_edge_servers=10, continuous_action=False):
        super().__init__()
        
        self.num_edge_servers = num_edge_servers
        self.continuous_action = continuous_action
        
        # Action space
        if continuous_action:
            self.action_space = spaces.Box(
                low=0, high=1, shape=(num_edge_servers,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(num_edge_servers)
        
        # Observation space
        self.observation_space = spaces.Dict({
            'task_size': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'server_speeds': spaces.Box(low=np.zeros(num_edge_servers), high=np.ones(num_edge_servers), dtype=np.float32),
            'server_loads': spaces.Box(low=np.zeros(num_edge_servers), high=np.ones(num_edge_servers), dtype=np.float32),
            'network_conditions': spaces.Box(low=np.zeros(num_edge_servers), high=np.ones(num_edge_servers), dtype=np.float32),
            'server_distances': spaces.Box(low=np.zeros(num_edge_servers), high=np.ones(num_edge_servers), dtype=np.float32)
        })
        
        # Initialize server characteristics with more distinct values
        self.server_speeds = np.random.uniform(0.6, 1.0, num_edge_servers)
        self.server_distances = np.random.uniform(0.1, 0.9, num_edge_servers)
        self.bandwidth_up = np.random.uniform(0.5, 1.0, num_edge_servers)
        self.bandwidth_down = np.random.uniform(0.6, 1.0, num_edge_servers)
        
        # Scaling factors for various latency components
        self.uplink_scale = 0.6
        self.prop_scale = 0.04
        self.downlink_scale = 0.5
        self.queue_factor = 0.9
        
        # Track history for normalization
        self.latency_history = deque(maxlen=100)
        self.prev_fluctuation = np.zeros(num_edge_servers)
        
        self.reset()
    
    def reset(self):
        """Reset environment state with new task and updated loads/conditions"""
        self.current_task_size = np.random.uniform(0.2, 0.8)
        self.server_loads = np.random.uniform(0.1, 0.4, self.num_edge_servers)
        self.network_conditions = np.random.uniform(0.7, 1.0, self.num_edge_servers)
        
        # Calculate initial effective speed
        self.effective_speeds = self.server_speeds * (1 - 0.8 * self.server_loads)
        
        # Generate rankings of servers
        self.speed_ranks = self.get_server_ranks(self.effective_speeds, reverse=True)
        self.load_ranks = self.get_server_ranks(self.server_loads)
        self.distance_ranks = self.get_server_ranks(self.server_distances)
        self.network_ranks = self.get_server_ranks(self.network_conditions, reverse=True)
        
        return self._get_observation()
    
    def get_server_ranks(self, values, reverse=False):
        """Generate server rankings based on a metric"""
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
        result_size = self.current_task_size * 0.05
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
            std_latency = np.std(self.latency_history) + 0.1
            latency_score = (total_latency - avg_latency) / std_latency
            latency_reward = -1.0 - np.clip(latency_score, -1.0, 1.0)  # Range: -2.0 to 0.0
        else:
            latency_reward = -total_latency / 2.0
        
        # 2. Server selection component (increased weight)
        # Calculate a composite rank considering multiple factors
        composite_rank = (
            self.speed_ranks[selected_server] * 0.4 + 
            self.load_ranks[selected_server] * 0.3 + 
            self.distance_ranks[selected_server] * 0.15 + 
            self.network_ranks[selected_server] * 0.15
        )
        
        normalized_rank = composite_rank / self.num_edge_servers
        
        # Higher reward for selecting better servers (0.0 to 0.8)
        selection_reward = 0.8 * (1.0 - normalized_rank)
        
        # 3. Load balancing component
        load_variance = np.var(self.server_loads)
        load_balance_reward = 0.2 * (1.0 - min(load_variance * 4, 1.0))
        
        # 4. Optimal server bonus
        effective_speeds = self.server_speeds * (1 - 0.8 * self.server_loads)
        optimal_server = np.argmax(effective_speeds)
        
        if selected_server == optimal_server:
            optimal_bonus = 0.4  # Bonus for choosing the optimal server
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
            'optimal_bonus': optimal_bonus
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
                # Linear decay for more stability
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


class PositionalEncoding(nn.Module):
    """Improved positional encoding for transformer"""
    def __init__(self, d_model, max_seq_length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerComponent(nn.Module):
    """Improved Transformer for temporal patterns with better attention mechanism"""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Add attention-based pooling
        self.attention_pooling = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_length, input_dim]
        
        # Project input to d_model dimension
        x = self.input_projection(x)
        x = self.input_norm(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        output = self.transformer(x, mask)
        
        # Apply attention-based pooling
        attention_weights = self.attention_pooling(output)
        weighted_output = output * attention_weights
        pooled_output = weighted_output.sum(dim=1)
        
        return pooled_output


class GNNComponent(nn.Module):
    """Improved GNN for server relationships with multi-head attention"""
    def __init__(self, node_features, hidden_dim=64, num_heads=2, dropout=0.1):
        super().__init__()
        
        # More expressive GAT layers
        self.conv1 = GATConv(node_features, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Edge attention
        self.edge_attention = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Node attention for pooling
        self.node_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=0)
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        # First GAT layer with normalization
        x1 = self.conv1(x, edge_index)
        x1 = F.elu(self.ln1(x1))
        
        # Second GAT layer with residual connection
        x2 = self.conv2(x1, edge_index)
        x2 = self.ln2(x2)
        x2 = F.elu(x2 + x1)  # Residual connection
        
        # Apply node-level attention for weighted pooling
        attention_weights = self.node_attention(x2)
        weighted_nodes = x2 * attention_weights
        pooled = weighted_nodes.sum(dim=0)
        
        return pooled


class VAEComponent(nn.Module):
    """Improved VAE for state distributions with better regularization"""
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32, dropout=0.1):
        super().__init__()
        
        # Enhanced encoder with residual connections and dropout
        self.encoder_layer1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_ln1 = nn.LayerNorm(hidden_dim)
        self.encoder_drop1 = nn.Dropout(dropout)
        
        self.encoder_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_ln2 = nn.LayerNorm(hidden_dim)
        self.encoder_drop2 = nn.Dropout(dropout)
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Enhanced decoder with residual connections and dropout
        self.decoder_layer1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_ln1 = nn.LayerNorm(hidden_dim)
        self.decoder_drop1 = nn.Dropout(dropout)
        
        self.decoder_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_ln2 = nn.LayerNorm(hidden_dim)
        self.decoder_drop2 = nn.Dropout(dropout)
        
        self.decoder_out = nn.Linear(hidden_dim, input_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using proper scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def encode(self, x):
        h1 = F.elu(self.encoder_ln1(self.encoder_layer1(x)))
        h1 = self.encoder_drop1(h1)
        
        h2 = F.elu(self.encoder_ln2(self.encoder_layer2(h1)))
        h2 = self.encoder_drop2(h2)
        h2 = h2 + h1  # Residual connection
        
        return self.fc_mu(h2), self.fc_logvar(h2)
    
    def reparameterize(self, mu, logvar):
        # More stable reparameterization
        std = torch.exp(0.5 * torch.clamp(logvar, -20, 2))
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h1 = F.elu(self.decoder_ln1(self.decoder_layer1(z)))
        h1 = self.decoder_drop1(h1)
        
        h2 = F.elu(self.decoder_ln2(self.decoder_layer2(h1)))
        h2 = self.decoder_drop2(h2)
        h2 = h2 + h1  # Residual connection
        
        return self.decoder_out(h2)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class CrossAttention(nn.Module):
    """Improved cross-attention module for feature fusion with multi-head attention"""
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Project inputs
        q = self.query_proj(query)  # [batch_size, hidden_dim]
        k = self.key_proj(key)      # [batch_size, hidden_dim]
        v = self.value_proj(value)  # [batch_size, hidden_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, 1, head_dim]
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, 1, head_dim]
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, 1, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, heads, 1, 1]
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.matmul(attention, v)  # [batch, heads, 1, head_dim]
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_dim)  # [batch, 1, hidden_dim]
        output = self.out_proj(output).squeeze(1)  # [batch, hidden_dim]
        
        return output


class DuelingQNetwork(nn.Module):
    """Enhanced Dueling Q-Network with better initialization and normalization"""
    def __init__(self, state_dim, action_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        
        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
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
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using proper scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, state):
        shared = self.shared_net(state)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        
        # Combine value and advantage (dueling architecture)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class SACComponent(nn.Module):
    """Improved SAC with better initialization and regularization"""
    def __init__(self, state_dim, action_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        
        # Q networks (using dueling architecture)
        self.q1 = DuelingQNetwork(state_dim, action_dim, hidden_dim, dropout)
        self.q2 = DuelingQNetwork(state_dim, action_dim, hidden_dim, dropout)
        
        # Policy network with improved architecture
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using proper scaling"""
        # Initialize policy network with smaller weights for more stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize head layers with smaller values
        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.xavier_uniform_(self.log_std_head.weight, gain=0.01)
        nn.init.constant_(self.log_std_head.bias, -1)  # Initialize to small std
        
    def forward(self, state):
        # State shape: [batch_size, state_dim]
        policy_features = self.policy_net(state)
        mean = self.mean_head(policy_features)
        
        # Constrain log_std for numerical stability
        log_std = torch.clamp(self.log_std_head(policy_features), -20, 2)
        
        q1_out = self.q1(state)
        q2_out = self.q2(state)
        
        return mean, log_std, q1_out, q2_out


class MECHybridSystem(nn.Module):
    """Improved hybrid system combining Transformer, GNN, VAE and SAC"""
    def __init__(self, state_dim, action_dim, num_servers=10, seq_length=10, 
                 hidden_dim=128, latent_dim=32, dropout=0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_servers = num_servers
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize improved components
        self.transformer = TransformerComponent(state_dim, hidden_dim, dropout=dropout)
        self.gnn = GNNComponent(4, hidden_dim//2, dropout=dropout)  # 4 features per server
        self.vae = VAEComponent(state_dim, hidden_dim, latent_dim, dropout=dropout)
        
        # Projection layers with normalization
        self.spatial_proj = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.vae_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Cross-attention fusion with improved mechanism
        self.temporal_to_spatial_attn = CrossAttention(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        self.temporal_to_vae_attn = CrossAttention(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        self.spatial_to_vae_attn = CrossAttention(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        
        # Final fusion layer with better regularization
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Initialize SAC component
        self.sac = SACComponent(hidden_dim, action_dim, dropout=dropout)
        
        # Add task-specific embedding
        self.task_embedding = nn.Embedding(10, hidden_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights using proper scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                
    def _process_batch_states(self, states):
        """Process a batch of states for GNN and VAE with better error handling"""
        if not isinstance(states, list):
            states = [states]
        batch_size = len(states)
        
        # Prepare tensors for each feature
        try:
            task_sizes = torch.stack([
                torch.tensor(s['task_size'], dtype=torch.float32, device=self.device)
                for s in states
            ])  # [batch_size, 1]
            
            server_speeds = torch.stack([
                torch.tensor(s['server_speeds'], dtype=torch.float32, device=self.device)
                for s in states
            ])  # [batch_size, num_servers]
            
            server_loads = torch.stack([
                torch.tensor(s['server_loads'], dtype=torch.float32, device=self.device)
                for s in states
            ])  # [batch_size, num_servers]
            
            network_conditions = torch.stack([
                torch.tensor(s['network_conditions'], dtype=torch.float32, device=self.device)
                for s in states
            ])  # [batch_size, num_servers]
            
            server_distances = torch.stack([
                torch.tensor(s['server_distances'], dtype=torch.float32, device=self.device)
                for s in states
            ])  # [batch_size, num_servers]
            
            # Create node features for GNN
            node_features = torch.stack([
                server_speeds,
                server_loads,
                network_conditions,
                server_distances
            ], dim=2)  # [batch_size, num_servers, 4]
            
            # Create state tensor for VAE
            state_tensor = torch.cat([
                task_sizes,
                server_speeds,
                server_loads,
                network_conditions,
                server_distances
            ], dim=1)  # [batch_size, state_dim]
            
            # Create edge features (distances between servers)
            edge_features = []
            for i in range(batch_size):
                edges = []
                for src in range(self.num_servers):
                    for dst in range(self.num_servers):
                        if src != dst:
                            edges.append(server_distances[i, dst].item())
                edge_features.append(edges)
            edge_features = torch.tensor(edge_features, dtype=torch.float32, device=self.device)
            
            return node_features, state_tensor, edge_features, batch_size
            
        except Exception as e:
            print(f"Error processing states: {e}")
            # Create default empty tensors as fallback
            node_features = torch.zeros((batch_size, self.num_servers, 4), device=self.device)
            state_tensor = torch.zeros((batch_size, self.state_dim), device=self.device)
            edge_features = torch.zeros((batch_size, self.num_servers * (self.num_servers - 1)), device=self.device)
            return node_features, state_tensor, edge_features, batch_size
        
    def forward(self, state_sequence, current_states):
        """Forward pass with improved error handling and component integration"""
        # Get batch size and ensure proper dimensions
        batch_size = state_sequence.size(0)
        
        # Process temporal patterns with transformer
        temporal_features = self.transformer(state_sequence)  # [batch_size, hidden_dim]
        
        # Process current states
        node_features, state_tensor, edge_features, _ = self._process_batch_states(current_states)
        
        # Create edges (fully connected) - shared across batch
        edge_index = []
        for i in range(self.num_servers):
            for j in range(self.num_servers):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t()
        
        # Process each batch element through GNN
        spatial_features_list = []
        for i in range(batch_size):
            batch_nodes = node_features[i]  # [num_servers, 4]
            batch_edges = edge_features[i].reshape(-1, 1)  # [num_edges, 1]
            gnn_output = self.gnn(batch_nodes, edge_index, batch_edges)  # [hidden_dim//2]
            spatial_features_list.append(gnn_output)
        
        spatial_features = torch.stack(spatial_features_list)  # [batch_size, hidden_dim//2]
        
        # Project spatial features to hidden_dim
        spatial_features = self.spatial_proj(spatial_features)  # [batch_size, hidden_dim]
        
        # Process state distribution with VAE
        vae_output, mu, logvar = self.vae(state_tensor)  # vae_output: [batch_size, state_dim]
        
        # Project VAE output to hidden_dim
        vae_features = self.vae_proj(vae_output)  # [batch_size, hidden_dim]
        
        # Apply cross-attention for feature fusion with residual connections
        enhanced_temporal = temporal_features + \
                          self.temporal_to_spatial_attn(temporal_features, spatial_features, spatial_features) + \
                          self.temporal_to_vae_attn(temporal_features, vae_features, vae_features)
        
        enhanced_spatial = spatial_features + \
                         self.temporal_to_spatial_attn(spatial_features, temporal_features, temporal_features) + \
                         self.spatial_to_vae_attn(spatial_features, vae_features, vae_features)
        
        enhanced_vae = vae_features + \
                     self.temporal_to_vae_attn(vae_features, temporal_features, temporal_features) + \
                     self.spatial_to_vae_attn(vae_features, spatial_features, spatial_features)
        
        # Combine features
        combined_features = torch.cat([
            enhanced_temporal,
            enhanced_spatial,
            enhanced_vae
        ], dim=1)  # [batch_size, hidden_dim * 3]
        
        # Fuse features
        fused_features = self.fusion(combined_features)  # [batch_size, hidden_dim]
        
        # Get action distribution and Q-values from SAC
        mean, log_std, q1, q2 = self.sac(fused_features)
        
        return {
            'policy': (mean, log_std),
            'q_values': (q1, q2),
            'vae_params': (mu, logvar),
            'fused_features': fused_features
        }


class PrioritizedReplayBuffer:
    """Enhanced Prioritized Experience Replay buffer with better error handling and stability"""
    def __init__(self, capacity, seq_length, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.capacity = capacity
        self.seq_length = seq_length
        self.position = 0
        self.size = 0
        
        # PER parameters
        self.alpha = alpha  # how much prioritization to use
        self.beta = beta_start  # importance-sampling correction
        self.beta_start = beta_start
        self.beta_frames = beta_frames  # frames over which to anneal beta to 1
        self.max_priority = 1.0  # max priority at start
        
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory with deep copying to avoid reference issues"""
        # Make deep copies of state dictionaries
        state_copy = {
            'task_size': np.array(state['task_size']),
            'server_speeds': np.array(state['server_speeds']),
            'server_loads': np.array(state['server_loads']),
            'network_conditions': np.array(state['network_conditions']),
            'server_distances': np.array(state['server_distances'])
        }
        
        next_state_copy = {
            'task_size': np.array(next_state['task_size']),
            'server_speeds': np.array(next_state['server_speeds']),
            'server_loads': np.array(next_state['server_loads']),
            'network_conditions': np.array(next_state['network_conditions']),
            'server_distances': np.array(next_state['server_distances'])
        }
        
        # Store transition with max priority
        if len(self.buffer) < self.capacity:
            self.buffer.append((state_copy, action, reward, next_state_copy, done))
        else:
            self.buffer[self.position] = (state_copy, action, reward, next_state_copy, done)
            
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample a batch of experiences with priorities"""
        # Ensure enough samples for sequences
        valid_idx = len(self.buffer) - self.seq_length
        if valid_idx < batch_size:
            return None
        
        # Update beta parameter
        self.beta = min(1.0, self.beta + (1.0 - self.beta_start) / self.beta_frames)
        
        # Get valid priorities
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        # Compute sampling probabilities
        # Add small constant to prevent zero probabilities
        p = (priorities[:valid_idx] + 1e-6) ** self.alpha
        p = p / p.sum()
        
        # Sample indices
        indices = np.random.choice(valid_idx, batch_size, p=p)
        
        # Calculate importance sampling weights
        weights = (valid_idx * p[indices]) ** (-self.beta)
        weights = weights / (weights.max() + 1e-10)  # Normalize and prevent division by zero
        weights = torch.FloatTensor(weights)
        
        batch_state_seqs = []
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        for start_idx in indices:
            # Get sequence
            state_seq = [self.buffer[i][0] for i in range(start_idx, start_idx + self.seq_length)]
            transition = self.buffer[start_idx + self.seq_length - 1]
            
            batch_state_seqs.append(state_seq)
            batch_states.append(transition[0])
            batch_actions.append(transition[1])
            batch_rewards.append(transition[2])
            batch_next_states.append(transition[3])
            batch_dones.append(transition[4])
        
        return (batch_state_seqs, batch_states, batch_actions, 
                batch_rewards, batch_next_states, batch_dones, indices, weights)
    
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions with better error handling"""
        try:
            for i, idx in enumerate(indices):
                # Ensure we're storing a simple scalar value
                if isinstance(priorities, np.ndarray):
                    # Handle arrays of any dimension by taking mean if needed
                    if priorities.size > len(indices):
                        # Multi-value per index - take mean
                        p_value = float(np.mean(priorities[i]))
                    else:
                        # One value per index
                        p_value = float(priorities[i])
                else:
                    # Try direct conversion
                    p_value = float(priorities[i])
                
                # Constrain priority value to prevent extreme values
                p_value = np.clip(p_value, 1e-6, 100.0)
                    
                self.priorities[idx] = p_value
                self.max_priority = max(self.max_priority, p_value)
        except Exception as e:
            print(f"Error updating priorities: {e}")

    def __len__(self):
        return self.size


class MECHybridAgent:
    """Enhanced Hybrid agent with improved training stability and exploration strategies"""
    def __init__(self, state_dim, action_dim, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize hybrid system
        self.model = MECHybridSystem(state_dim, action_dim).to(device)
        self.target_model = MECHybridSystem(state_dim, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Initialize optimizers with learning rate scheduling
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=5e-5)
        
        # Initialize alpha (SAC temperature parameter)
        self.target_entropy = -action_dim  # Negative action dimension
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        
        # Initialize replay buffer with prioritized experience replay
        self.replay_buffer = PrioritizedReplayBuffer(100000, seq_length=10)
        
        # Initialize episode buffer for sequence building
        self.episode_buffer = []
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005  # Target network update rate
        self.batch_size = 64  # Smaller batch size for more frequent updates
        self.min_replay_size = 2000  # Larger minimum buffer
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.15  # Higher min epsilon
        self.epsilon_decay = 0.9995  # Much slower decay
        
        # Cyclical epsilon parameters
        self.use_cyclical_epsilon = True
        self.cycle_length = 200  # Longer cycles
        self.cycle_min = 0.15
        self.cycle_max = 0.4
        
        # Add n-step returns
        self.n_steps = 3
        self.n_step_buffer = []
        
        # Component weighting
        self.transformer_weight = 1.0
        self.gnn_weight = 1.0
        self.vae_weight = 1.0
        
        # Training statistics
        self.training_stats = {
            'critic_losses': [],
            'actor_losses': [],
            'vae_losses': [],
            'alpha_losses': [],
            'mean_q_values': [],
            'td_errors': []
        }
        
    def _prepare_sequence(self, state_sequence):
        """Convert state sequence to tensor with better error handling"""
        try:
            sequence = []
            for state in state_sequence:
                # Flatten dictionary state into tensor
                state_components = [
                    state['task_size'],
                    state['server_speeds'],
                    state['server_loads'],
                    state['network_conditions'],
                    state['server_distances']
                ]
                # Concatenate all components into a single tensor
                state_tensor = torch.cat([
                    torch.tensor(comp, dtype=torch.float32, device=self.device).flatten() 
                    for comp in state_components
                ])
                sequence.append(state_tensor)
            return torch.stack(sequence)
        except Exception as e:
            print(f"Error preparing sequence: {e}")
            # Return dummy tensor as fallback
            return torch.zeros((len(state_sequence), self.state_dim), device=self.device)
    
    def _update_epsilon(self, episode):
        """Update epsilon with improved cyclical schedule"""
        if self.use_cyclical_epsilon and episode > 300:  # Start cyclical after more initial exploration
            # Cyclical epsilon strategy after initial learning phase
            cycle_position = episode % self.cycle_length
            cycle_ratio = cycle_position / self.cycle_length
            
            # Cosine annealing within cycle for smoother transitions
            self.epsilon = self.cycle_min + 0.5 * (self.cycle_max - self.cycle_min) * \
                          (1 + np.cos(cycle_ratio * np.pi))
        else:
            # Standard epsilon decay for initial learning
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def select_action(self, state, evaluate=False):
        """Select action with improved exploration strategies"""
        # Add state to episode buffer
        self.episode_buffer.append(state)
        if len(self.episode_buffer) > 10:
            self.episode_buffer.pop(0)
        
        # Create state sequence
        if len(self.episode_buffer) < 10:
            padding = [self.episode_buffer[0]] * (10 - len(self.episode_buffer))
            state_sequence = padding + self.episode_buffer
        else:
            state_sequence = self.episode_buffer
        
        # Convert sequence to tensor
        state_sequence_tensor = self._prepare_sequence(state_sequence)
        state_sequence_tensor = state_sequence_tensor.unsqueeze(0)  # Add batch dimension
        
        # Epsilon-greedy exploration with improved strategies
        if not evaluate and random.random() < self.epsilon:
            exploration_method = random.random()
            
            if exploration_method < 0.6:  # Standard random action
                return random.randrange(self.action_dim)
            elif exploration_method < 0.85:  # Server load-aware exploration
                # Select servers with lower load more often
                server_loads = np.array(state['server_loads'])
                # Get effective speeds considering both load and speed
                effective_speeds = np.array(state['server_speeds']) * (1 - 0.8 * server_loads)
                # Normalize to probabilities (higher effective speed = higher probability)
                probabilities = effective_speeds / (effective_speeds.sum() + 1e-10)
                return np.random.choice(self.action_dim, p=probabilities)
            else:  # Noisy network exploration - Thompson sampling style
                with torch.no_grad():
                    # Clone the model to avoid affecting training
                    temp_model = MECHybridSystem(self.state_dim, self.action_dim).to(self.device)
                    temp_model.load_state_dict(self.model.state_dict())
                    
                    # Add noise only to the final layers
                    for name, param in temp_model.named_parameters():
                        if 'head' in name or 'fusion' in name:
                            param.data += torch.randn_like(param) * 0.01
                    
                    # Get action from noisy model
                    outputs = temp_model(state_sequence_tensor, [state])
                    mean, _ = outputs['policy']
                    action = torch.argmax(mean).item()
                    
                    # Clean up temporary model
                    del temp_model
                    
                    return action
        
        # Standard action selection for evaluation or exploitation
        with torch.no_grad():
            outputs = self.model(state_sequence_tensor, [state])
            mean, log_std = outputs['policy']
            
            if evaluate:
                # For evaluation, just take the highest probability action
                return torch.argmax(mean).item()
            
            # For non-evaluation exploitation, sample from the distribution
            std = log_std.exp()
            dist = Normal(mean, std)
            action = dist.rsample()
            
            # Convert continuous action to discrete by taking argmax
            return torch.argmax(action).item()
    
    def _compute_n_step_returns(self, rewards, next_value, dones):
        """Compute n-step returns for more accurate target values"""
        returns = []
        R = next_value
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)
        return returns
    
    def train(self):
        """Train the agent with improved stability and better gradient handling"""
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        # Sample from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None
        
        state_seqs, states, actions, rewards, next_states, dones, indices, weights = batch
        
        # Prepare tensors
        state_seqs_tensor = torch.stack([
            self._prepare_sequence(seq) for seq in state_seqs
        ]).to(self.device)
        
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        weights_tensor = weights.to(self.device)  # Importance sampling weights
        
        # Get current model outputs
        outputs = self.model(state_seqs_tensor, states)
        mean, log_std = outputs['policy']
        q1, q2 = outputs['q_values']
        mu, logvar = outputs['vae_params']
        
        # Select q-values for the taken actions
        q1_selected = q1.gather(1, actions_tensor.unsqueeze(1))
        q2_selected = q2.gather(1, actions_tensor.unsqueeze(1))
        
        # Get target model outputs for next states
        with torch.no_grad():
            next_outputs = self.target_model(state_seqs_tensor, next_states)
            next_mean, next_log_std = next_outputs['policy']
            next_q1, next_q2 = next_outputs['q_values']
            
            # Sample actions from next policy
            next_std = next_log_std.exp()
            next_dist = Normal(next_mean, next_std)
            next_actions = next_dist.rsample()
            next_log_probs = next_dist.log_prob(next_actions)
            
            # Get q-values for sampled actions
            alpha = self.log_alpha.exp().detach()
            next_q_target = torch.min(next_q1, next_q2)
            next_q_target = next_q_target - alpha * next_log_probs.sum(1, keepdim=True)
            
            # Compute target Q values with n-step returns if available
            if hasattr(self, 'n_step_buffer') and len(self.n_step_buffer) > 0:
                # Compute n-step returns
                n_step_rewards = [transition[2] for transition in self.n_step_buffer[-self.n_steps:]]
                n_step_dones = [transition[4] for transition in self.n_step_buffer[-self.n_steps:]]
                n_step_target = self._compute_n_step_returns(n_step_rewards, next_q_target, n_step_dones)[-1]
                target_q = n_step_target
            else:
                # Standard one-step TD target
                target_q = rewards_tensor.unsqueeze(1) + \
                          (1 - dones_tensor.unsqueeze(1)) * self.gamma * next_q_target
        
        # Compute losses
        # Critic loss (TD error) with importance sampling weights
        q1_loss = (weights_tensor.unsqueeze(1) * F.smooth_l1_loss(q1_selected, target_q, reduction='none')).mean()
        q2_loss = (weights_tensor.unsqueeze(1) * F.smooth_l1_loss(q2_selected, target_q, reduction='none')).mean()
        critic_loss = q1_loss + q2_loss
        
        # Actor loss (policy gradient)
        std = log_std.exp()
        dist = Normal(mean, std)
        actions_sampled = dist.rsample()
        log_probs = dist.log_prob(actions_sampled)
        
        alpha = self.log_alpha.exp()
        min_q = torch.min(
            self.model.sac.q1(outputs['fused_features']),
            self.model.sac.q2(outputs['fused_features'])
        )
        actor_loss = (alpha * log_probs - min_q).mean()
        
        # Alpha loss (temperature adjustment)
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy).detach()).mean()
        
        # VAE loss (reconstruction + KL divergence) with annealing
        vae_recon_loss = F.mse_loss(outputs['vae_params'][0], torch.zeros_like(outputs['vae_params'][0]))
        kl_weight = min(1.0, len(self.replay_buffer) / 10000)  # Gradually increase KL weight
        vae_kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * kl_weight
        vae_loss = vae_recon_loss + 0.1 * vae_kl_loss
        
        # Total loss with component weighting
        total_loss = (
            self.transformer_weight * critic_loss + 
            actor_loss + 
            self.vae_weight * 0.1 * vae_loss
        )
        
        # Compute TD errors for priority updates
        with torch.no_grad():
            # Mean absolute TD error
            td_errors = torch.abs(q1_selected - target_q).mean(dim=1).cpu().numpy()
        
        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update target network with polyak averaging
        self._soft_update_target_network()
        
        # Update priorities in the replay buffer
        self.replay_buffer.update_priorities(indices, td_errors + 1e-6)
        
        # Track training statistics
        self.training_stats['critic_losses'].append(critic_loss.item())
        self.training_stats['actor_losses'].append(actor_loss.item())
        self.training_stats['vae_losses'].append(vae_loss.item())
        self.training_stats['alpha_losses'].append(alpha_loss.item())
        self.training_stats['mean_q_values'].append(q1.mean().item())
        self.training_stats['td_errors'].append(td_errors.mean())
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'vae_loss': vae_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item(),
            'mean_td_error': td_errors.mean()
        }
    
    def _soft_update_target_network(self):
        """Soft update target network parameters using polyak averaging"""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
            
    def adjust_component_weights(self, transformer_weight=None, gnn_weight=None, vae_weight=None):
        """Adjust the weights of different components for ablation studies"""
        if transformer_weight is not None:
            self.transformer_weight = transformer_weight
        if gnn_weight is not None:
            self.gnn_weight = gnn_weight
        if vae_weight is not None:
            self.vae_weight = vae_weight
        print(f"Component weights set to: Transformer={self.transformer_weight}, GNN={self.gnn_weight}, VAE={self.vae_weight}")
    
    def save_model(self, path):
        """Save the model and training state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'log_alpha': self.log_alpha,
            'training_stats': self.training_stats
        }, path)
        
    def load_model(self, path, weights_only=True):
        """Load the model and training state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        
        if not weights_only:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.log_alpha = checkpoint['log_alpha']
            self.training_stats = checkpoint['training_stats']


def train_mec_hybrid():
    """Training loop with improved monitoring and early stopping logic"""
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
    agent = MECHybridAgent(state_size, env.num_edge_servers)
    
    # Create directory for saving models
    os.makedirs('models', exist_ok=True)
    
    # Training parameters
    num_episodes = 2000
    max_steps = 100
    eval_frequency = 25  # More frequent evaluation
    
    # Early stopping parameters with more patience
    early_stop_patience = 20  # More patience
    best_reward = -float('inf')
    no_improvement_count = 0
    
    # Metrics tracking
    metrics = {
        'rewards': [],
        'critic_losses': [],
        'actor_losses': [],
        'vae_losses': [],
        'alpha_losses': [],
        'alphas': [],
        'latencies': [],
        'server_loads': [],
        'avg_rewards': [],
        'epsilon_values': [],
        'eval_rewards': [],
        'eval_latencies': []
    }
    
    # Create evaluation function
    def evaluate_agent(eval_episodes=10):
        """Evaluate agent performance without exploration"""
        eval_rewards = []
        eval_latencies = []
        server_selections = []
        
        # Store current epsilon and temporarily set to 0 for evaluation
        current_epsilon = agent.epsilon
        agent.epsilon = 0.0
        
        for _ in range(eval_episodes):
            state = env.reset()
            episode_reward = 0
            episode_latencies = []
            episode_servers = []
            
            # Clear episode buffer for clean evaluation
            agent.episode_buffer = []
            
            for step in range(max_steps):
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, info = env.step(action)
                
                episode_latencies.append(info['total_latency'])
                episode_servers.append(action)
                state = next_state
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            eval_latencies.append(np.mean(episode_latencies))
            server_selections.append(episode_servers)
        
        # Restore original epsilon
        agent.epsilon = current_epsilon
        
        # Calculate server diversity (how many different servers were used)
        server_diversity = np.mean([len(set(servers)) for servers in server_selections])
        
        return np.mean(eval_rewards), np.mean(eval_latencies), server_diversity
    
    # Start timer
    start_time = time.time()
    
    # Dynamic component adjustment tracking
    adjustment_history = []
    
    # Optional: Pre-training of VAE for latent space learning
    print("Starting training loop...")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        episode_latencies = []
        episode_servers = []
        
        # Clear episode buffer at start of episode
        agent.episode_buffer = []
        agent.n_step_buffer = []  # Clear n-step buffer
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Store in n-step buffer
            agent.n_step_buffer.append((state, action, reward, next_state, done))
            if len(agent.n_step_buffer) > agent.n_steps:
                agent.n_step_buffer.pop(0)
            
            # Train agent
            losses = agent.train()
            
            if losses is not None:
                episode_losses.append(losses)
            
            episode_latencies.append(info['total_latency'])
            episode_servers.append(action)
            state = next_state
            episode_reward += reward
        
        # Update exploration rate with improved schedule
        agent._update_epsilon(episode)
        
        # Update metrics
        metrics['rewards'].append(episode_reward)
        metrics['latencies'].append(np.mean(episode_latencies))
        metrics['server_loads'].append(np.mean(state['server_loads']))
        metrics['epsilon_values'].append(agent.epsilon)
        
        if episode_losses:
            metrics['critic_losses'].append(np.mean([loss['critic_loss'] for loss in episode_losses]))
            metrics['actor_losses'].append(np.mean([loss['actor_loss'] for loss in episode_losses]))
            metrics['vae_losses'].append(np.mean([loss['vae_loss'] for loss in episode_losses]))
            metrics['alpha_losses'].append(np.mean([loss['alpha_loss'] for loss in episode_losses]))
            metrics['alphas'].append(np.mean([loss['alpha'] for loss in episode_losses]))
        
        # Calculate running average
        if episode >= eval_frequency:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            metrics['avg_rewards'].append(avg_reward)
        
        # Step the scheduler every 5 episodes for more stable learning rate decay
        if episode % 5 == 0:
            agent.scheduler.step()
        
        # Print progress and evaluate
        if episode % eval_frequency == 0:
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            avg_reward = np.mean(metrics['rewards'][-min(eval_frequency, len(metrics['rewards'])):])
            avg_latency = np.mean(metrics['latencies'][-min(eval_frequency, len(metrics['latencies'])):])
            
            # Run evaluation
            eval_reward, eval_latency, server_diversity = evaluate_agent()
            metrics['eval_rewards'].append(eval_reward)
            metrics['eval_latencies'].append(eval_latency)
            
            # Calculate server selection statistics
            server_counts = np.bincount(episode_servers, minlength=env.num_edge_servers)
            most_used_server = np.argmax(server_counts)
            most_used_pct = server_counts[most_used_server] / len(episode_servers) * 100
            
            # Get current learning rate
            current_lr = agent.optimizer.param_groups[0]['lr']
            
            print(f"Episode {episode}/{num_episodes} [{time_str}], "
                  f"Train Reward: {avg_reward:.2f}, "
                  f"Train Latency: {avg_latency:.2f}, "
                  f"Eval Reward: {eval_reward:.2f}, "
                  f"Eval Latency: {eval_latency:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"LR: {current_lr:.6f}, "
                  f"Server Diversity: {server_diversity:.1f}/{env.num_edge_servers}")
            
            # Check for early stopping with more patience
            if eval_reward > best_reward:
                improvement = eval_reward - best_reward
                best_reward = eval_reward
                no_improvement_count = 0
                
                # Save best model
                agent.save_model("models/best_mec_hybrid.pt")
                print(f"New best model saved with reward: {best_reward:.2f} (improvement: {improvement:.2f})")
            else:
                no_improvement_count += 1
                current_deficit = best_reward - eval_reward
                print(f"No improvement for {no_improvement_count} evaluations (deficit: {current_deficit:.2f})")
                
            # Dynamic adjustment based on performance trends
            if no_improvement_count >= early_stop_patience // 2 and no_improvement_count < early_stop_patience:
                print("Performance plateaued - trying component weight adjustment")
                # Try adjusting component weights
                new_transformer_weight = 1.0 + 0.1 * (random.random() - 0.5)  # 0.95 to 1.05
                new_gnn_weight = 1.0 + 0.2 * random.random()  # 1.0 to 1.2
                new_vae_weight = 1.0 + 0.2 * random.random()  # 1.0 to 1.2
                
                agent.adjust_component_weights(
                    transformer_weight=new_transformer_weight,
                    gnn_weight=new_gnn_weight,
                    vae_weight=new_vae_weight
                )
                
                adjustment_history.append({
                    'episode': episode,
                    'transformer_weight': new_transformer_weight,
                    'gnn_weight': new_gnn_weight,
                    'vae_weight': new_vae_weight
                })
            
            if no_improvement_count >= early_stop_patience:
                print(f"Early stopping at episode {episode}")
                break
            
            # Save checkpoint every 100 episodes
            if episode % 100 == 0 and episode > 0:
                agent.save_model(f"models/mec_hybrid_ep{episode}.pt")
                print(f"Checkpoint saved at episode {episode}")
    
    # Load best model for final evaluation
    print("Loading best model for final evaluation...")
    agent.load_model("models/best_mec_hybrid.pt")
    
    # Final extensive evaluation
    final_reward, final_latency, final_diversity = evaluate_agent(eval_episodes=50)
    print(f"Final evaluation - Reward: {final_reward:.2f}, Latency: {final_latency:.2f}, "
          f"Server Diversity: {final_diversity:.1f}/{env.num_edge_servers}")
    
    # Save all metrics
    np.save("models/training_metrics.npy", metrics)
    print("Training metrics saved")
    
    return agent, metrics


if __name__ == "__main__":
    agent, metrics = train_mec_hybrid()