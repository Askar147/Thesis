import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.distributions import Normal
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import random
import os
import time
import json
from collections import deque
from datetime import datetime

# Import the MEC environment and models
from mec_environment import MECEnvironment

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer component"""
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
    """Transformer for temporal patterns"""
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
        
        # Attention-based pooling
        self.attention_pooling = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x, mask=None):
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
    """GNN for server relationships"""
    def __init__(self, node_features, hidden_dim=64, num_heads=2, dropout=0.1):
        super().__init__()
        
        # GAT layers
        self.conv1 = GATConv(node_features, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
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
    """VAE for state distributions"""
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32, dropout=0.1):
        super().__init__()
        
        # Encoder with residual connections
        self.encoder_layer1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_ln1 = nn.LayerNorm(hidden_dim)
        self.encoder_drop1 = nn.Dropout(dropout)
        
        self.encoder_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_ln2 = nn.LayerNorm(hidden_dim)
        self.encoder_drop2 = nn.Dropout(dropout)
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder with residual connections
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
        # Stable reparameterization
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
    """Cross-attention module for feature fusion"""
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
    """Dueling Q-Network with better initialization and normalization"""
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
    """Soft Actor-Critic with better initialization and regularization"""
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

def flatten_observation(obs, num_mvs, num_edge_servers):
    """
    Flatten the observation dictionary for a specific MV index into a vector
    suitable for the network input
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

class MECHybridSystem(nn.Module):
    """Hybrid system combining Transformer, GNN, VAE and SAC for MEC task offloading"""
    def __init__(self, state_dim, action_dim, num_servers=10, seq_length=10, 
                 hidden_dim=128, latent_dim=32, dropout=0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_servers = num_servers
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
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
        
        # Cross-attention fusion
        self.temporal_to_spatial_attn = CrossAttention(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        self.temporal_to_vae_attn = CrossAttention(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        self.spatial_to_vae_attn = CrossAttention(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        
        # Fusion layer
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
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _process_batch_states(self, states):
        """Process a batch of states for GNN and VAE"""
        if not isinstance(states, list):
            states = [states]
        batch_size = len(states)
        
        # Process GNN features
        node_features_list = []
        state_tensors = []
        
        for state in states:
            # Extract server features for GNN
            server_features = []
            server_features.append(state['es_compute_capacity'].flatten())
            server_features.append(state['es_queue_length'].flatten())
            
            # Get network characteristics for first MV
            channel_gain = state['channel_gain'][0, :]
            network_delay = state['network_delay'][0, :]
            
            # Create node features for GNN
            node_feat = np.stack([
                state['es_compute_capacity'].flatten(),
                state['es_queue_length'].flatten(),
                channel_gain,
                network_delay
            ], axis=1)  # [num_servers, 4]
            
            node_features_list.append(torch.FloatTensor(node_feat).to(self.device))
            
            # Create state tensor for VAE
            flattened_state = np.concatenate([
                state['task_size'][0],
                state['required_cpu_cycles'][0],
                state['task_deadline'][0],
                state['mv_compute_capacity'][0],
                state['mv_local_queue'][0],
                state['mv_energy'][0],
                state['es_compute_capacity'].flatten(),
                state['es_queue_length'].flatten(),
                channel_gain,
                network_delay
            ])
            
            state_tensors.append(torch.FloatTensor(flattened_state).to(self.device))
        
        # Create edge indices for GNN (fully connected between servers)
        edge_index = []
        for i in range(self.num_servers):
            for j in range(self.num_servers):
                if i != j:
                    edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t()
        
        return node_features_list, state_tensors, edge_index, batch_size
        
    def forward(self, state_sequence, current_states):
        """Forward pass with component integration"""
        # Get batch size and ensure proper dimensions
        batch_size = state_sequence.size(0)
        
        # Process temporal patterns with transformer
        temporal_features = self.transformer(state_sequence)  # [batch_size, hidden_dim]
        
        # Process current states
        node_features_list, state_tensors, edge_index, _ = self._process_batch_states(current_states)
        
        # Process each batch element through GNN
        spatial_features_list = []
        for i in range(batch_size):
            batch_nodes = node_features_list[i]  # [num_servers, 4]
            gnn_output = self.gnn(batch_nodes, edge_index)  # [hidden_dim//2]
            spatial_features_list.append(gnn_output)
        
        spatial_features = torch.stack(spatial_features_list)  # [batch_size, hidden_dim//2]
        
        # Project spatial features to hidden_dim
        spatial_features = self.spatial_proj(spatial_features)  # [batch_size, hidden_dim]
        
        # Process state distribution with VAE
        state_tensor_batch = torch.stack(state_tensors)  # [batch_size, actual_state_dim]
        
        # ADD THIS BLOCK TO FIX VAE DIMENSION MISMATCH
        # Create new VAE with correct input dimension if needed
        if not hasattr(self, 'vae_adjusted'):
            print("Creating new VAE with correct input dimension")
            actual_state_dim = state_tensor_batch.shape[1]
            self.vae = VAEComponent(actual_state_dim, self.hidden_dim, 32, dropout=0.1).to(self.device)
            self.vae_proj = nn.Sequential(
                nn.Linear(actual_state_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim)
            ).to(self.device)
            self.vae_adjusted = True
        
        vae_output, mu, logvar = self.vae(state_tensor_batch)
        
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
    """Prioritized Experience Replay buffer"""
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
        # Make deep copies of state dictionaries to avoid reference issues
        state_copy = {}
        for key, value in state.items():
            state_copy[key] = np.array(value)
            
        next_state_copy = {}
        for key, value in next_state.items():
            next_state_copy[key] = np.array(value)
        
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
        """Update priorities of sampled transitions"""
        try:
            for i, idx in enumerate(indices):
                # Convert priority to a simple scalar value
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
    """Hybrid agent for MEC task offloading"""
    def __init__(self, state_dim, action_dim, num_edge_servers, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_edge_servers = num_edge_servers
        
        print(f"Using device: {self.device}")
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        
        # Initialize hybrid system
        self.model = MECHybridSystem(state_dim, action_dim, num_servers=num_edge_servers).to(device)
        self.target_model = MECHybridSystem(state_dim, action_dim, num_servers=num_edge_servers).to(device)
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
        self.batch_size = 64
        self.min_replay_size = 5000  # Larger minimum buffer
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay_steps = 2500
        self.current_step = 0
        
        # Cyclical epsilon parameters
        self.use_cyclical_epsilon = True
        self.cycle_length = 200
        self.cycle_min = 0.1
        self.cycle_max = 0.3
        
        # Add n-step returns
        self.n_steps = 3
        self.n_step_buffer = []
        
        # Component weighting
        self.transformer_weight = 1.0
        self.gnn_weight = 1.0
        self.vae_weight = 0.5  # Lower weight for VAE component
        
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
        """Convert state sequence to tensor for transformer input"""
        try:
            sequence = []
            for state in state_sequence:
                # Flatten each state observation in the sequence
                flattened = flatten_observation(state, 1, self.num_edge_servers)
                sequence.append(torch.FloatTensor(flattened))
            
            return torch.stack(sequence).to(self.device)
        except Exception as e:
            print(f"Error preparing sequence: {e}")
            # Return dummy tensor as fallback
            return torch.zeros((len(state_sequence), self.state_dim), device=self.device)
    
    def _update_epsilon(self, episode):
        """Update epsilon with cyclical schedule"""
        if self.use_cyclical_epsilon and episode > 300:  # Start cyclical after initial exploration
            # Cyclical epsilon strategy after initial learning phase
            cycle_position = episode % self.cycle_length
            cycle_ratio = cycle_position / self.cycle_length
            
            # Cosine annealing within cycle for smoother transitions
            self.epsilon = self.cycle_min + 0.5 * (self.cycle_max - self.cycle_min) * \
                         (1 + np.cos(cycle_ratio * np.pi))
        else:
            # Standard epsilon decay for initial learning
            if self.current_step < self.epsilon_decay_steps:
                self.epsilon = 1.0 - (1.0 - self.epsilon_min) * (self.current_step / self.epsilon_decay_steps)
            else:
                self.epsilon = self.epsilon_min
    
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
            # Three different exploration strategies
            exploration_method = random.random()
            
            if exploration_method < 0.7:  # Standard random action
                return random.randrange(self.action_dim)
            elif exploration_method < 0.9:  # Server load-aware exploration
                try:
                    # Select servers with lower load more often
                    server_loads = state['es_queue_length'].flatten()
                    server_speeds = state['es_compute_capacity'].flatten()
                    
                    # Get effective speeds considering both load and speed
                    # Add a small positive value to ensure it's never negative
                    effective_speeds = server_speeds * np.maximum(0.01, 1 - 0.8 * server_loads)
                    # Add local processing option (with medium priority)
                    effective_speeds = np.append(np.mean(effective_speeds), effective_speeds)
                    
                    # Ensure all values are positive
                    effective_speeds = np.maximum(0.001, effective_speeds)
                    
                    # Normalize to probabilities
                    probabilities = effective_speeds / np.sum(effective_speeds)
                    
                    # Debug
                    if np.any(probabilities < 0) or not np.all(np.isfinite(probabilities)):
                        print("Invalid probabilities detected, using uniform distribution")
                        return random.randrange(self.action_dim)
                    
                    return np.random.choice(self.action_dim, p=probabilities)
                except Exception as e:
                    print(f"Error in server-aware exploration: {e}")
                    return random.randrange(self.action_dim)
            else:  # Network-aware exploration
                try:
                    # Select based on network conditions
                    mv_idx = 0  # First mobile vehicle
                    channel_gains = np.array([state['channel_gain'][mv_idx]])
                    network_delays = np.array([state['network_delay'][mv_idx]])
                    
                    # Combine gains and delays into network quality metric
                    # Ensure no division by zero and positive values
                    network_quality = channel_gains / (network_delays + 1e-10)
                    network_quality = np.maximum(0.001, network_quality)  # Ensure positive
                    
                    # Add local processing option (with medium priority)
                    network_quality = np.append(np.mean(network_quality), network_quality.flatten())
                    
                    # Normalize to probabilities
                    sum_quality = np.sum(network_quality)
                    if sum_quality <= 0:
                        print("Invalid network quality sum, using uniform distribution")
                        return random.randrange(self.action_dim)
                        
                    probabilities = network_quality / sum_quality
                    
                    # Verify probabilities are valid
                    if np.any(probabilities < 0) or not np.all(np.isfinite(probabilities)):
                        print("Invalid probabilities detected, using uniform distribution")
                        return random.randrange(self.action_dim)
                    
                    return np.random.choice(self.action_dim, p=probabilities)
                except Exception as e:
                    print(f"Error in network-aware exploration: {e}")
                    return random.randrange(self.action_dim)
        
        # Standard action selection for evaluation or exploitation
        with torch.no_grad():
            outputs = self.model(state_sequence_tensor, [state])
            mean, log_std = outputs['policy']
            
            if evaluate:
                # For evaluation, just take the highest probability action
                return torch.argmax(mean).item()
            
            # For exploitation, add some noise for exploration
            std = log_std.exp()
            dist = Normal(mean, std)
            action_probs = F.softmax(dist.rsample(), dim=0)
            
            # Get discrete action with highest probability
            return torch.argmax(action_probs).item()
    
    def _compute_n_step_returns(self, rewards, next_value, dones):
        """Compute n-step returns for more accurate target values"""
        returns = []
        R = next_value
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)
        return returns
    
    def train(self):
        """Train the agent with multi-component architecture"""
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        # Sample from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None
        
        state_seqs, states, actions, rewards, next_states, dones, indices, weights = batch
        
        # Prepare state sequences for transformer
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
            
            # REPLACE THESE LINES:
            # next_q_target = next_q_target - alpha * next_log_probs.sum(1, keepdim=True)
            # target_q = rewards_tensor.unsqueeze(1) + (1 - dones_tensor.unsqueeze(1)) * self.gamma * next_q_target
            
            # WITH THESE LINES:
            # Fix: Ensure next_q_target has shape [batch_size, 1]
            if next_q_target.dim() > 2 or next_q_target.shape[1] > 1:
                # Take the appropriate action value or average if needed
                next_q_target = next_q_target.mean(dim=1, keepdim=True)
            
            # Also ensure next_log_probs is properly shaped
            entropy_term = next_log_probs
            if entropy_term.dim() > 2:
                entropy_term = entropy_term.sum(dim=1, keepdim=True)
            elif entropy_term.dim() == 2 and entropy_term.shape[1] > 1:
                entropy_term = entropy_term.sum(dim=1, keepdim=True)
            
            # Subtract entropy term with corrected shape
            next_q_target = next_q_target - alpha * entropy_term
            
            # Compute target Q values - ensure this has shape [batch_size, 1]
            if hasattr(self, 'n_step_buffer') and len(self.n_step_buffer) >= self.n_steps:
                # Compute n-step returns
                n_step_rewards = [transition[2] for transition in self.n_step_buffer]
                n_step_dones = [transition[4] for transition in self.n_step_buffer]
                n_step_target = self._compute_n_step_returns(n_step_rewards, next_q_target, n_step_dones)[-1]
                target_q = n_step_target
            else:
                # Standard one-step TD target
                target_q = rewards_tensor.unsqueeze(1) + \
                        (1 - dones_tensor.unsqueeze(1)) * self.gamma * next_q_target
        
        # Add this debug print to check shapes
        
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
        
        # VAE loss (reconstruction + KL divergence)
        vae_recon_loss = F.mse_loss(outputs['vae_params'][0], torch.zeros_like(outputs['vae_params'][0]))
        kl_weight = min(1.0, len(self.replay_buffer) / 50000)  # Gradually increase KL weight
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
        
        # Increment step counter
        self.current_step += 1
        
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
        print(f"Model saved to {path}")
        
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
        
        print(f"Model loaded from {path}")

def train_mec_hybrid(env_config):
    """Training loop for MEC environment with Hybrid architecture"""
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
    
    # Initialize Hybrid agent
    agent = MECHybridAgent(state_size, action_size, env.num_edge_servers)
    
    # Training parameters
    num_episodes = env_config['num_episodes']
    max_steps = env_config['max_steps']
    eval_frequency = 50
    
    # Early stopping parameters
    early_stop_patience = 20
    best_reward = -float('inf')
    no_improvement_count = 0
    
    # Enhanced metrics tracking
    metrics = {
        'rewards': [],
        'critic_losses': [],
        'actor_losses': [],
        'vae_losses': [],
        'alpha_losses': [],
        'alphas': [],
        'latencies': [],
        'task_completion_rates': [],
        'processing_times': [],
        'energy_consumption': [],
        'avg_rewards': [],
        'epsilon_values': [],
        'eval_rewards': [],
        'eval_latencies': []
    }
    
    # Create directories for saving models and results
    os.makedirs('hybrid_models', exist_ok=True)
    
    # Pre-fill replay buffer with random episodes
    print("Pre-filling replay buffer with random experiences...")
    episode_count = 0
    
    while len(agent.replay_buffer) < agent.min_replay_size:
        episode_count += 1
        obs = env.reset()
        
        # Clear episode buffer at start of episode
        agent.episode_buffer = []
        agent.n_step_buffer = []
        
        for step in range(max_steps):
            # Select random action
            action = random.randrange(action_size)
            
            # Take step
            next_obs, rewards, done, info = env.step(action)
            reward = rewards[0]  # First MV
            
            # Store transition
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            
            # Store in n-step buffer
            agent.n_step_buffer.append((obs, action, reward, next_obs, done))
            if len(agent.n_step_buffer) > agent.n_steps:
                agent.n_step_buffer.pop(0)
            
            obs = next_obs
            
            if done:
                break
                
        # Print progress occasionally
        if episode_count % 10 == 0:
            print(f"Pre-filling buffer: {len(agent.replay_buffer)}/{agent.min_replay_size} transitions (episode {episode_count})")
    
    print(f"Replay buffer filled with {len(agent.replay_buffer)} transitions. Starting training...")
    
    # Start training timer
    start_time = time.time()
    
    # Dynamic component adjustment tracking
    adjustment_history = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_losses = []
        episode_completion_rates = []
        episode_processing_times = []
        episode_energy = []
        episode_latencies = []
        
        # Clear episode buffer at start of episode
        agent.episode_buffer = []
        agent.n_step_buffer = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(obs)
            next_obs, rewards, done, info = env.step(action)
            
            # Get reward for the specific MV we're training
            reward = rewards[0]  # First MV
            
            # Store transition and n-step information
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            
            # Store in n-step buffer
            agent.n_step_buffer.append((obs, action, reward, next_obs, done))
            if len(agent.n_step_buffer) > agent.n_steps:
                agent.n_step_buffer.pop(0)
            
            # Train agent
            losses = agent.train()
            
            if losses is not None:
                episode_losses.append(losses)
            
            # Track metrics
            episode_completion_rates.append(info['task_completion_rate'])
            episode_processing_times.append(info['avg_processing_time'])
            episode_energy.append(info['avg_energy_consumption'])
            episode_latencies.append(info['avg_processing_time'])  # Using processing time as latency
            
            obs = next_obs
            episode_reward += reward
            
            if done:
                break
        
        # Update exploration rate with improved schedule
        agent._update_epsilon(episode)
        
        # Step the scheduler every 5 episodes
        if episode % 5 == 0:
            agent.scheduler.step()
        
        # Store metrics
        metrics['rewards'].append(episode_reward)
        metrics['task_completion_rates'].append(np.mean(episode_completion_rates))
        metrics['processing_times'].append(np.mean(episode_processing_times))
        metrics['energy_consumption'].append(np.mean(episode_energy))
        metrics['latencies'].append(np.mean(episode_latencies))
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
        
        # Print progress and evaluate
        if episode % eval_frequency == 0:
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            # Calculate metrics
            avg_reward = np.mean(metrics['rewards'][-min(eval_frequency, len(metrics['rewards'])):])
            avg_latency = np.mean(metrics['latencies'][-min(eval_frequency, len(metrics['latencies'])):])
            avg_completion = np.mean(metrics['task_completion_rates'][-min(eval_frequency, len(metrics['task_completion_rates'])):])
            avg_energy = np.mean(metrics['energy_consumption'][-min(eval_frequency, len(metrics['energy_consumption'])):])
            
            # Run evaluation
            eval_reward, eval_latency, eval_completion, eval_energy = evaluate_agent(agent, env)
            metrics['eval_rewards'].append(eval_reward)
            metrics['eval_latencies'].append(eval_latency)
            
            # Get current learning rate
            current_lr = agent.optimizer.param_groups[0]['lr']
            
            print(f"Episode {episode}/{num_episodes} [{time_str}], "
                  f"Train Reward: {avg_reward:.2f}, "
                  f"Eval Reward: {eval_reward:.2f}, "
                  f"Completion Rate: {avg_completion:.2f}, "
                  f"Latency: {avg_latency:.4f}, "
                  f"Energy: {avg_energy:.4f}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"LR: {current_lr:.6f}")
            
            # Check for early stopping
            if eval_reward > best_reward:
                improvement = eval_reward - best_reward
                best_reward = eval_reward
                no_improvement_count = 0
                
                # Save best model
                agent.save_model(f"hybrid_models/best_mec_hybrid.pt")
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
                new_vae_weight = 0.5 + 0.5 * random.random()  # 0.5 to 1.0
                
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
                agent.save_model(f"hybrid_models/mec_hybrid_ep{episode}.pt")
                print(f"Checkpoint saved at episode {episode}")
                
                # Save current metrics
                save_training_results(metrics, env_config)
    
    # Save final model and metrics
    agent.save_model("hybrid_models/final_mec_hybrid.pt")
    save_training_results(metrics, env_config)
    
    # Final evaluation
    print("Loading best model for final evaluation...")
    agent.load_model("hybrid_models/best_mec_hybrid.pt")
    final_reward, final_latency, final_completion, final_energy = evaluate_agent(agent, env, num_episodes=50)
    
    print(f"Final evaluation - Reward: {final_reward:.2f}, Latency: {final_latency:.4f}, "
          f"Completion Rate: {final_completion:.2f}, Energy: {final_energy:.4f}")
    
    return agent, metrics

def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate agent performance without exploration"""
    total_rewards = []
    total_latencies = []
    total_completion_rates = []
    total_energy_consumption = []
    
    # Store current epsilon and set to 0 for deterministic evaluation
    current_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_latencies = []
        episode_completion_rates = []
        episode_energy = []
        
        # Clear episode buffer for clean evaluation
        agent.episode_buffer = []
        
        for step in range(100):  # Use fixed max steps for evaluation
            # Select action deterministically
            action = agent.select_action(obs, evaluate=True)
            
            # Take action
            next_obs, rewards, done, info = env.step(action)
            
            # Record metrics
            episode_latencies.append(info['avg_processing_time'])
            episode_completion_rates.append(info['task_completion_rate'])
            episode_energy.append(info['avg_energy_consumption'])
            
            # Update state and reward
            episode_reward += rewards[0]  # First MV
            obs = next_obs
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        total_latencies.append(np.mean(episode_latencies))
        total_completion_rates.append(np.mean(episode_completion_rates))
        total_energy_consumption.append(np.mean(episode_energy))
    
    # Restore original epsilon
    agent.epsilon = current_epsilon
    
    # Return average metrics
    return (
        np.mean(total_rewards),
        np.mean(total_latencies),
        np.mean(total_completion_rates),
        np.mean(total_energy_consumption)
    )

def save_training_results(metrics, env_config=None, save_dir="hybrid_results"):
    """Save training metrics and plots to specified directory"""
    # Create results directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save configuration if provided
    if env_config:
        config_filename = os.path.join(save_dir, f"hybrid_config_{timestamp}.json")
        with open(config_filename, 'w') as f:
            json.dump(env_config, f, indent=4)
    
    # Save metrics to JSON file
    metrics_filename = os.path.join(save_dir, f"hybrid_metrics_{timestamp}.json")
    json_metrics = {
        'rewards': [float(r) for r in metrics['rewards']],
        'critic_losses': [float(l) for l in metrics.get('critic_losses', [])],
        'actor_losses': [float(l) for l in metrics.get('actor_losses', [])],
        'vae_losses': [float(l) for l in metrics.get('vae_losses', [])],
        'alpha_losses': [float(l) for l in metrics.get('alpha_losses', [])],
        'alphas': [float(a) for a in metrics.get('alphas', [])],
        'task_completion_rates': [float(t) for t in metrics.get('task_completion_rates', [])],
        'processing_times': [float(p) for p in metrics.get('processing_times', [])],
        'energy_consumption': [float(e) for e in metrics.get('energy_consumption', [])],
        'latencies': [float(l) for l in metrics.get('latencies', [])],
        'epsilon_values': [float(e) for e in metrics.get('epsilon_values', [])],
        'eval_rewards': [float(r) for r in metrics.get('eval_rewards', [])],
        'eval_latencies': [float(l) for l in metrics.get('eval_latencies', [])]
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
    
    # Add evaluation rewards if available
    if 'eval_rewards' in metrics and metrics['eval_rewards']:
        eval_episodes = range(0, len(metrics['rewards']), 50)  # Every 50 episodes
        plt.plot(eval_episodes[:len(metrics['eval_rewards'])], 
                 metrics['eval_rewards'], 'go-', label='Evaluation Rewards')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Training and Evaluation Rewards')
    plt.grid(True)
    
    # Plot component losses
    plt.subplot(2, 2, 2)
    if 'critic_losses' in metrics and metrics['critic_losses']:
        plt.plot(metrics['critic_losses'], label='Critic Loss')
    if 'actor_losses' in metrics and metrics['actor_losses']:
        plt.plot(metrics['actor_losses'], label='Actor Loss')
    if 'vae_losses' in metrics and metrics['vae_losses']:
        plt.plot(metrics['vae_losses'], label='VAE Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Component Losses')
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
        plt.title('Average Latency')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = os.path.join(save_dir, f"hybrid_training_plot_{timestamp}.png")
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
    if 'epsilon_values' in metrics and metrics['epsilon_values']:
        plt.plot(metrics['epsilon_values'], label='Epsilon')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate (Epsilon)')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save second plot
    plot2_filename = os.path.join(save_dir, f"hybrid_training_plot2_{timestamp}.png")
    plt.savefig(plot2_filename, dpi=300, bbox_inches='tight')
    
    # Third figure: Temperature parameter and component weights
    plt.figure(figsize=(15, 5))
    
    # Plot alpha (temperature parameter)
    plt.subplot(1, 2, 1)
    if 'alphas' in metrics and metrics['alphas']:
        plt.plot(metrics['alphas'], label='Alpha')
        plt.xlabel('Episode')
        plt.ylabel('Alpha')
        plt.title('Temperature Parameter (Alpha)')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save third plot
    plot3_filename = os.path.join(save_dir, f"hybrid_training_plot3_{timestamp}.png")
    plt.savefig(plot3_filename, dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print(f"Results saved to directory: {save_dir}")
    print(f"Metrics saved as: hybrid_metrics_{timestamp}.json")
    print(f"Plots saved as: hybrid_training_plot_{timestamp}.png, hybrid_training_plot2_{timestamp}.png, and hybrid_training_plot3_{timestamp}.png")

def compare_all_models(ddqn_results_dir, gnn_results_dir, transformer_results_dir, hybrid_results_dir, 
                      save_dir="model_comparison"):
    """Compare all four model performances: DDQN, GNN, Transformer, and Hybrid"""
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
    
    # Load Hybrid metrics
    hybrid_metrics_file = None
    for file in os.listdir(hybrid_results_dir):
        if file.startswith("hybrid_metrics_") and file.endswith(".json"):
            hybrid_metrics_file = os.path.join(hybrid_results_dir, file)
            break
    
    if not hybrid_metrics_file:
        print("No Hybrid metrics found.")
        return
        
    with open(hybrid_metrics_file, 'r') as f:
        hybrid_metrics = json.load(f)
    
    # Create comparison plots
    plt.figure(figsize=(20, 15))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(ddqn_metrics['rewards'], alpha=0.3, label='DDQN')
    plt.plot(gnn_metrics['rewards'], alpha=0.3, label='GNN')
    plt.plot(transformer_metrics['rewards'], alpha=0.3, label='Transformer')
    plt.plot(hybrid_metrics['rewards'], alpha=0.3, label='Hybrid')
    
    # Calculate and plot moving averages
    window_size = 20
    for name, metrics in [('DDQN', ddqn_metrics), ('GNN', gnn_metrics), 
                         ('Transformer', transformer_metrics), ('Hybrid', hybrid_metrics)]:
        if len(metrics['rewards']) > window_size:
            moving_avg = np.convolve(metrics['rewards'], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            x_avg = np.arange(window_size-1, len(metrics['rewards']))
            plt.plot(x_avg, moving_avg, label=f'{name} Moving Avg', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Reward Comparison')
    plt.grid(True)
    
    # Plot task completion rates
    plt.subplot(2, 2, 2)
    plt.plot(ddqn_metrics['task_completion_rates'], alpha=0.3, label='DDQN')
    plt.plot(gnn_metrics['task_completion_rates'], alpha=0.3, label='GNN')
    plt.plot(transformer_metrics['task_completion_rates'], alpha=0.3, label='Transformer')
    plt.plot(hybrid_metrics['task_completion_rates'], alpha=0.3, label='Hybrid')
    
    # Calculate and plot moving averages
    for name, metrics in [('DDQN', ddqn_metrics), ('GNN', gnn_metrics), 
                         ('Transformer', transformer_metrics), ('Hybrid', hybrid_metrics)]:
        if len(metrics['task_completion_rates']) > window_size:
            moving_avg = np.convolve(metrics['task_completion_rates'], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            x_avg = np.arange(window_size-1, len(metrics['task_completion_rates']))
            plt.plot(x_avg, moving_avg, label=f'{name} Moving Avg', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Completion Rate')
    plt.legend()
    plt.title('Task Completion Rate Comparison')
    plt.grid(True)
    
    # Plot processing times
    plt.subplot(2, 2, 3)
    plt.plot(ddqn_metrics['processing_times'], alpha=0.3, label='DDQN')
    plt.plot(gnn_metrics['processing_times'], alpha=0.3, label='GNN')
    plt.plot(transformer_metrics['processing_times'], alpha=0.3, label='Transformer')
    if 'latencies' in hybrid_metrics:
        plt.plot(hybrid_metrics['latencies'], alpha=0.3, label='Hybrid')
    else:
        plt.plot(hybrid_metrics['processing_times'], alpha=0.3, label='Hybrid')
    
    # Calculate and plot moving averages
    for name, metrics, key in [('DDQN', ddqn_metrics, 'processing_times'), 
                              ('GNN', gnn_metrics, 'processing_times'), 
                              ('Transformer', transformer_metrics, 'processing_times'), 
                              ('Hybrid', hybrid_metrics, 'latencies' if 'latencies' in hybrid_metrics else 'processing_times')]:
        if len(metrics[key]) > window_size:
            moving_avg = np.convolve(metrics[key], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            x_avg = np.arange(window_size-1, len(metrics[key]))
            plt.plot(x_avg, moving_avg, label=f'{name} Moving Avg', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Processing Time / Latency')
    plt.legend()
    plt.title('Processing Time / Latency Comparison')
    plt.grid(True)
    
    # Plot energy consumption
    plt.subplot(2, 2, 4)
    plt.plot(ddqn_metrics['energy_consumption'], alpha=0.3, label='DDQN')
    plt.plot(gnn_metrics['energy_consumption'], alpha=0.3, label='GNN')
    plt.plot(transformer_metrics['energy_consumption'], alpha=0.3, label='Transformer')
    plt.plot(hybrid_metrics['energy_consumption'], alpha=0.3, label='Hybrid')
    
    # Calculate and plot moving averages
    for name, metrics in [('DDQN', ddqn_metrics), ('GNN', gnn_metrics), 
                         ('Transformer', transformer_metrics), ('Hybrid', hybrid_metrics)]:
        if len(metrics['energy_consumption']) > window_size:
            moving_avg = np.convolve(metrics['energy_consumption'], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            x_avg = np.arange(window_size-1, len(metrics['energy_consumption']))
            plt.plot(x_avg, moving_avg, label=f'{name} Moving Avg', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Energy Consumption')
    plt.legend()
    plt.title('Energy Consumption Comparison')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_filename = os.path.join(save_dir, f"all_models_comparison_{timestamp}.png")
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Evaluation metrics comparison
    # Create a table of final performance metrics
    final_window = min(100, 
                      min(len(ddqn_metrics['rewards']), 
                         len(gnn_metrics['rewards']),
                         len(transformer_metrics['rewards']),
                         len(hybrid_metrics['rewards'])))
    
    # Get evaluation metrics if available
    eval_metrics = {}
    for name, metrics in [('DDQN', ddqn_metrics), ('GNN', gnn_metrics), 
                         ('Transformer', transformer_metrics), ('Hybrid', hybrid_metrics)]:
        if 'eval_rewards' in metrics and metrics['eval_rewards']:
            eval_metrics[name] = {
                'reward': metrics['eval_rewards'][-1],
                'latency': metrics['eval_latencies'][-1] if 'eval_latencies' in metrics else None
            }
    
    # Create a table of training metrics
    training_metrics = {
        'Model': ['DDQN', 'GNN', 'Transformer', 'Hybrid'],
        'Avg Reward': [
            np.mean(ddqn_metrics['rewards'][-final_window:]),
            np.mean(gnn_metrics['rewards'][-final_window:]),
            np.mean(transformer_metrics['rewards'][-final_window:]),
            np.mean(hybrid_metrics['rewards'][-final_window:])
        ],
        'Avg Completion Rate': [
            np.mean(ddqn_metrics['task_completion_rates'][-final_window:]),
            np.mean(gnn_metrics['task_completion_rates'][-final_window:]),
            np.mean(transformer_metrics['task_completion_rates'][-final_window:]),
            np.mean(hybrid_metrics['task_completion_rates'][-final_window:])
        ],
        'Avg Processing Time': [
            np.mean(ddqn_metrics['processing_times'][-final_window:]),
            np.mean(gnn_metrics['processing_times'][-final_window:]),
            np.mean(transformer_metrics['processing_times'][-final_window:]),
            np.mean(hybrid_metrics['latencies'][-final_window:] 
                   if 'latencies' in hybrid_metrics else 
                   hybrid_metrics['processing_times'][-final_window:])
        ],
        'Avg Energy Consumption': [
            np.mean(ddqn_metrics['energy_consumption'][-final_window:]),
            np.mean(gnn_metrics['energy_consumption'][-final_window:]),
            np.mean(transformer_metrics['energy_consumption'][-final_window:]),
            np.mean(hybrid_metrics['energy_consumption'][-final_window:])
        ]
    }
    
    # Add evaluation metrics if available
    if eval_metrics:
        training_metrics['Eval Reward'] = [
            eval_metrics.get('DDQN', {}).get('reward', float('nan')),
            eval_metrics.get('GNN', {}).get('reward', float('nan')),
            eval_metrics.get('Transformer', {}).get('reward', float('nan')),
            eval_metrics.get('Hybrid', {}).get('reward', float('nan'))
        ]
        training_metrics['Eval Latency'] = [
            eval_metrics.get('DDQN', {}).get('latency', float('nan')),
            eval_metrics.get('GNN', {}).get('latency', float('nan')),
            eval_metrics.get('Transformer', {}).get('latency', float('nan')),
            eval_metrics.get('Hybrid', {}).get('latency', float('nan'))
        ]
    
    # Save comparison table to JSON
    table_filename = os.path.join(save_dir, f"all_models_comparison_table_{timestamp}.json")
    with open(table_filename, 'w') as f:
        json.dump(training_metrics, f, indent=4)
    
    # Print comparison table
    print("\nPerformance Comparison (last 100 episodes):")
    headers = list(training_metrics.keys())
    row_format = "{:>15}" * len(headers)
    print(row_format.format(*headers))
    print("-" * 15 * len(headers))
    
    for i in range(4):  # Four models
        row = [training_metrics[header][i] for header in headers]
        # Format floats to 4 decimal places
        formatted_row = [item if isinstance(item, str) else f"{item:.4f}" for item in row]
        print(row_format.format(*formatted_row))
    
    print(f"\nComparison results saved to {comparison_filename} and {table_filename}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    
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
    run_dir = os.path.join("hybrid_results", f"hybrid_run_{timestamp}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # Train the agent
    print(f"Starting training with configuration: {env_config}")
    agent, metrics = train_mec_hybrid(env_config)
    
    # Save results
    save_training_results(metrics, env_config, save_dir=run_dir)
    
    print(f"Training completed. Results saved to {run_dir}")
    
    # Optional: Compare with other models if available
    ddqn_results_dir = "results"  # Directory containing DDQN results
    gnn_results_dir = "gnn_results"  # Directory containing GNN results
    transformer_results_dir = "transformer_results"  # Directory containing Transformer results
    
    if os.path.exists(ddqn_results_dir) and os.path.exists(gnn_results_dir) and os.path.exists(transformer_results_dir):
        print("Comparing all model results...")
        compare_all_models(ddqn_results_dir, gnn_results_dir, transformer_results_dir, run_dir)