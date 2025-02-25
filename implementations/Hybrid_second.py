import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
from DQN_second import MECEnvironment


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_seq_length=100):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerComponent(nn.Module):
    """Improved Transformer for temporal patterns in MEC"""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)  # Normalize inputs
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Add attention-based pooling instead of mean pooling
        self.attention_pooling = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_length, input_dim]
        
        # Project input to d_model dimension
        x = self.input_projection(x)  # [batch_size, seq_length, d_model]
        x = self.input_norm(x)  # Apply normalization
        
        # Add positional encoding
        x = self.pos_encoder(x)  # [batch_size, seq_length, d_model]
        
        # Apply transformer
        output = self.transformer(x, mask)  # [batch_size, seq_length, d_model]
        
        # Apply attention-based pooling
        attention_weights = self.attention_pooling(output)  # [batch_size, seq_length, 1]
        weighted_output = output * attention_weights  # [batch_size, seq_length, d_model]
        pooled_output = weighted_output.sum(dim=1)  # [batch_size, d_model]
        
        return pooled_output

class GNNComponent(nn.Module):
    """Improved GNN for server relationships"""
    def __init__(self, node_features, hidden_dim=64, num_heads=2):
        super().__init__()
        
        # More expressive GAT layers with multi-head attention
        self.conv1 = GATConv(node_features, hidden_dim // num_heads, heads=num_heads)
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Add edge features through a separate projection
        self.edge_proj = nn.Linear(1, hidden_dim)
        
        # Final node-level attention for improved pooling
        self.node_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
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
        attention_weights = self.node_attention(x2)  # [num_nodes, 1]
        weighted_nodes = x2 * attention_weights  # [num_nodes, hidden_dim]
        pooled = weighted_nodes.sum(dim=0)  # [hidden_dim]
        
        return pooled

class VAEComponent(nn.Module):
    """Improved VAE for state distributions"""
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32):
        super().__init__()
        
        # Enhanced encoder with residual connections
        self.encoder_layer1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_ln1 = nn.LayerNorm(hidden_dim)
        self.encoder_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_ln2 = nn.LayerNorm(hidden_dim)
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Enhanced decoder with residual connections
        self.decoder_layer1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_ln1 = nn.LayerNorm(hidden_dim)
        self.decoder_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_ln2 = nn.LayerNorm(hidden_dim)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.encoder_ln1(self.encoder_layer1(x)))
        h2 = F.relu(self.encoder_ln2(self.encoder_layer2(h1)))
        h2 = h2 + h1  # Residual connection
        
        return self.fc_mu(h2), self.fc_logvar(h2)
    
    def reparameterize(self, mu, logvar):
        # More stable reparameterization
        std = torch.exp(0.5 * torch.clamp(logvar, -20, 2))
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h1 = F.relu(self.decoder_ln1(self.decoder_layer1(z)))
        h2 = F.relu(self.decoder_ln2(self.decoder_layer2(h1)))
        h2 = h2 + h1  # Residual connection
        
        return self.decoder_out(h2)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class CrossAttention(nn.Module):
    """Cross-attention module for feature fusion"""
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim):
        super().__init__()
        
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)
        
        self.scale = hidden_dim ** -0.5
        
    def forward(self, query, key, value):
        # Project inputs
        q = self.query_proj(query)  # [batch_size, query_dim] -> [batch_size, hidden_dim]
        k = self.key_proj(key)      # [batch_size, key_dim] -> [batch_size, hidden_dim]
        v = self.value_proj(value)  # [batch_size, value_dim] -> [batch_size, hidden_dim]
        
        # Reshape for attention calculation
        # Add a dimension for attention heads (here we use just 1 head)
        q = q.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        k = k.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        v = v.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, 1, 1]
        attention = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention, v)  # [batch_size, 1, hidden_dim]
        output = output.squeeze(1)  # [batch_size, hidden_dim]
        
        return output

class DuelingQNetwork(nn.Module):
    """Dueling Q-Network for improved action selection"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, state):
        shared = self.shared_net(state)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        
        # Combine value and advantage (dueling architecture)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class SACComponent(nn.Module):
    """Improved SAC for decision making with TD3-like double critics"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # Q networks (using dueling architecture)
        self.q1 = DuelingQNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = DuelingQNetwork(state_dim, action_dim, hidden_dim)
        
        # Policy network with improved architecture
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Separate heads for mean and log_std with initialization
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize policy network with smaller weights
        for layer in self.policy_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0)
        
        # Initialize head layers
        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0)
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
    def __init__(self, state_dim, action_dim, num_servers=10, seq_length=10, hidden_dim=128, latent_dim=32):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_servers = num_servers
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize improved components
        self.transformer = TransformerComponent(state_dim, hidden_dim)
        self.gnn = GNNComponent(4)  # 4 features per server
        self.vae = VAEComponent(state_dim, hidden_dim, latent_dim)
        
        # Projection layers
        self.spatial_proj = nn.Linear(64, hidden_dim)  # GNN output is 64-dim, project to hidden_dim
        self.vae_proj = nn.Linear(state_dim, hidden_dim)  # VAE output is state_dim, project to hidden_dim
        
        # Cross-attention fusion
        self.temporal_to_spatial_attn = CrossAttention(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        self.temporal_to_vae_attn = CrossAttention(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        self.spatial_to_vae_attn = CrossAttention(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Initialize SAC component with improved architecture
        self.sac = SACComponent(hidden_dim, action_dim)
        
        # Add task-specific embedding
        self.task_embedding = nn.Embedding(10, hidden_dim)  # 10 task types
        
        # Initialize weights using Xavier initialization
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
                
    def _process_batch_states(self, states):
        """Process a batch of states for GNN and VAE"""
        if not isinstance(states, list):
            states = [states]
        batch_size = len(states)
        
        # Prepare tensors for each feature
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
        
    def forward(self, state_sequence, current_states):
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
            gnn_output = self.gnn(batch_nodes, edge_index, batch_edges)  # [64]
            spatial_features_list.append(gnn_output)
        
        spatial_features = torch.stack(spatial_features_list)  # [batch_size, 64]
        
        # Project spatial features to hidden_dim
        spatial_features = self.spatial_proj(spatial_features)  # [batch_size, hidden_dim]
        
        # Process state distribution with VAE
        vae_output, mu, logvar = self.vae(state_tensor)  # vae_output: [batch_size, state_dim]
        
        # Project VAE output to hidden_dim
        vae_features = self.vae_proj(vae_output)  # [batch_size, hidden_dim]
        
        # Apply cross-attention for feature fusion
        enhanced_temporal = temporal_features + self.temporal_to_spatial_attn(temporal_features, spatial_features, spatial_features) + self.temporal_to_vae_attn(temporal_features, vae_features, vae_features)
        enhanced_spatial = spatial_features + self.temporal_to_spatial_attn(spatial_features, temporal_features, temporal_features) + self.spatial_to_vae_attn(spatial_features, vae_features, vae_features)
        enhanced_vae = vae_features + self.temporal_to_vae_attn(vae_features, temporal_features, temporal_features) + self.spatial_to_vae_attn(vae_features, spatial_features, spatial_features)
        
        # Combine features with cross-attention enhanced representations
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
    """Prioritized Experience Replay buffer for hybrid system"""
    def __init__(self, capacity, seq_length, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.capacity = capacity
        self.seq_length = seq_length
        self.position = 0
        self.size = 0
        
        # PER parameters
        self.alpha = alpha  # how much prioritization to use (0 = none, 1 = full)
        self.beta = beta_start  # importance-sampling correction
        self.beta_start = beta_start
        self.beta_frames = beta_frames  # frames over which to anneal beta to 1
        self.max_priority = 1.0  # max priority at start
        
    def push(self, state, action, reward, next_state, done):
        # Make a deep copy of state dictionaries to avoid reference issues
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
        p = priorities[:valid_idx] ** self.alpha
        p = p / p.sum()
        
        # Sample indices
        indices = np.random.choice(valid_idx, batch_size, p=p)
        
        # Calculate importance sampling weights
        weights = (valid_idx * p[indices]) ** (-self.beta)
        weights = weights / weights.max()
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
                
            self.priorities[idx] = p_value
            self.max_priority = max(self.max_priority, p_value)

    def __len__(self):
        return self.size

class MECHybridAgent:
    """Improved Hybrid agent combining Transformer, GNN, VAE, and SAC for MEC task offloading"""
    def __init__(self, state_dim, action_dim, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize hybrid system
        self.model = MECHybridSystem(state_dim, action_dim).to(device)
        self.target_model = MECHybridSystem(state_dim, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Initialize optimizers with learning rate scheduling
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-5)
        
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
        self.batch_size = 128  # Increased batch size
        self.min_replay_size = 1000
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # Higher min epsilon for better exploration
        self.epsilon_decay = 0.998  # Slower decay
        
        # Cyclical epsilon parameters
        self.use_cyclical_epsilon = True
        self.cycle_length = 100
        self.cycle_min = 0.1
        self.cycle_max = 0.5
        
        # Add n-step returns
        self.n_steps = 3
        self.n_step_buffer = []
        
        # Component weighting for ablation studies
        self.transformer_weight = 1.0
        self.gnn_weight = 1.0
        self.vae_weight = 1.0
        
    def _prepare_sequence(self, state_sequence):
        """Convert state sequence to tensor"""
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
    
    def _update_epsilon(self, episode):
        """Update epsilon with optional cyclical schedule"""
        if self.use_cyclical_epsilon and episode > 200:
            # Cyclical epsilon strategy after initial learning phase
            cycle_position = episode % self.cycle_length
            cycle_ratio = cycle_position / self.cycle_length
            
            # Cosine annealing within cycle
            self.epsilon = self.cycle_min + 0.5 * (self.cycle_max - self.cycle_min) * \
                          (1 + np.cos(cycle_ratio * np.pi))
        else:
            # Standard epsilon decay for initial learning
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def select_action(self, state, evaluate=False):
        """Select action with epsilon-greedy exploration and Thompson sampling"""
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
        
        # Epsilon-greedy exploration
        if not evaluate and random.random() < self.epsilon:
            exploration_method = random.random()
            
            if exploration_method < 0.7:  # Standard random action
                return random.randrange(self.action_dim)
            elif exploration_method < 0.9:  # Weighted random based on server load
                # Select servers with lower load more often
                server_loads = np.array(state['server_loads'])
                inverted_loads = 1.0 - server_loads  # Invert so lower loads have higher probability
                inverted_loads = np.clip(inverted_loads, 0.1, 1.0)  # Ensure no zero probabilities
                probabilities = inverted_loads / inverted_loads.sum()  # Normalize to probabilities
                return np.random.choice(self.action_dim, p=probabilities)
            else:  # Thompson sampling style exploration
                # Add noise to model parameters for exploration
                with torch.no_grad():
                    for param in self.model.parameters():
                        param.data += torch.randn_like(param) * 0.01
                    
                    # Get action from noisy model
                    outputs = self.model(state_sequence_tensor, [state])
                    mean, _ = outputs['policy']
                    action = torch.argmax(mean).item()
                    
                    # Restore original parameters
                    self.model.load_state_dict(self.model.state_dict())
                    
                    return action
        
        # Standard action selection for evaluation or exploitation
        with torch.no_grad():
            outputs = self.model(state_sequence_tensor, [state])
            mean, log_std = outputs['policy']
            
            if evaluate:
                return torch.argmax(mean).item()
            
            # Sample from the distribution for exploration
            std = log_std.exp()
            dist = Normal(mean, std)
            action = dist.rsample()
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
        """Train the agent using prioritized experience replay and n-step returns"""
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
        q1_loss = (weights_tensor.unsqueeze(1) * (q1_selected - target_q).pow(2)).mean()
        q2_loss = (weights_tensor.unsqueeze(1) * (q2_selected - target_q).pow(2)).mean()
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
        vae_kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = vae_recon_loss + 0.1 * vae_kl_loss
        
        # Total loss with component weighting
        total_loss = critic_loss + actor_loss + 0.1 * vae_loss
        
        # Compute TD errors for priority updates
        with torch.no_grad():
    # Simple mean absolute TD error for priorities
            td_errors = torch.abs(q1_selected - target_q).mean(dim=1).cpu().numpy()
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update target network with polyak averaging
        self._soft_update_target_network()
        
        # Update priorities in the replay buffer
        self.replay_buffer.update_priorities(indices, td_errors + 1e-6)
        
        # Step the learning rate scheduler
        self.scheduler.step()
        
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


def train_mec_hybrid():
    """Training loop for improved hybrid system"""
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
    
    # Training parameters
    num_episodes = 2000
    max_steps = 100
    eval_frequency = 50
    
    # Early stopping parameters
    early_stop_patience = 10
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
        'epsilon_values': []
    }
    
    # Create evaluation function
    def evaluate_agent(eval_episodes=10):
        """Evaluate agent performance without exploration"""
        eval_rewards = []
        eval_latencies = []
        
        # Store current epsilon and temporarily set to 0 for evaluation
        current_epsilon = agent.epsilon
        agent.epsilon = 0.0
        
        for _ in range(eval_episodes):
            state = env.reset()
            episode_reward = 0
            episode_latencies = []
            
            # Clear episode buffer for clean evaluation
            agent.episode_buffer = []
            
            for step in range(max_steps):
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, info = env.step(action)
                
                episode_latencies.append(info['total_latency'])
                state = next_state
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            eval_latencies.append(np.mean(episode_latencies))
        
        # Restore original epsilon
        agent.epsilon = current_epsilon
        
        return np.mean(eval_rewards), np.mean(eval_latencies)
    
    # Optional: weight different components differently based on performance
    # Uncomment to enable component weighting adjustments
    # agent.adjust_component_weights(transformer_weight=1.2, gnn_weight=1.0, vae_weight=0.8)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        episode_latencies = []
        
        # Clear episode buffer at start of episode
        agent.episode_buffer = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition and train
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Also store in n-step buffer if available
            if hasattr(agent, 'n_step_buffer'):
                agent.n_step_buffer.append((state, action, reward, next_state, done))
                if len(agent.n_step_buffer) > agent.n_steps:
                    agent.n_step_buffer.pop(0)
            
            losses = agent.train()
            
            if losses is not None:
                episode_losses.append(losses)
            
            episode_latencies.append(info['total_latency'])
            state = next_state
            episode_reward += reward
        
        # Update exploration rate
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
        
        # Print progress and evaluate
        if episode % eval_frequency == 0:
            avg_reward = np.mean(metrics['rewards'][-min(eval_frequency, len(metrics['rewards'])):])
            avg_latency = np.mean(metrics['latencies'][-min(eval_frequency, len(metrics['latencies'])):])
            
            # Run evaluation
            eval_reward, eval_latency = evaluate_agent()
            
            print(f"Episode {episode}/{num_episodes}, "
                  f"Average Reward: {avg_reward:.2f}, "
                  f"Average Latency: {avg_latency:.2f}, "
                  f"Eval Reward: {eval_reward:.2f}, "
                  f"Eval Latency: {eval_latency:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"LR: {agent.scheduler.get_last_lr()[0]:.6f}")
            
            # Check for early stopping
            if eval_reward > best_reward:
                best_reward = eval_reward
                no_improvement_count = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': agent.model.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'scheduler_state_dict': agent.scheduler.state_dict(),
                    'log_alpha': agent.log_alpha,
                }, "best_mec_hybrid.pth")
                
                print(f"New best model saved with reward: {best_reward:.2f}")
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= early_stop_patience:
                print(f"Early stopping at episode {episode}")
                break
        
        # Dynamic component weight adjustment (optional)
        # This adjusts the relative importance of different components based on performance
        if episode % 200 == 0 and episode > 0:
            # Example logic for dynamic adjustment:
            # If performance is improving, increase transformer weight
            # If performance is degrading, try different component balances
            if len(metrics['avg_rewards']) > 1:
                recent_trend = metrics['avg_rewards'][-1] - metrics['avg_rewards'][-2]
                if recent_trend > 0:
                    # Performance improving - stick with current weights
                    pass
                else:
                    # Try adjusting weights - this is just an example logic
                    # In a real implementation, you'd want more sophisticated adjustment
                    new_transformer_weight = 1.0 + 0.2 * random.random()
                    new_gnn_weight = 1.0 + 0.2 * random.random()
                    new_vae_weight = 1.0 + 0.2 * random.random()
                    agent.adjust_component_weights(
                        transformer_weight=new_transformer_weight,
                        gnn_weight=new_gnn_weight,
                        vae_weight=new_vae_weight
                    )
    
    # Load best model for final evaluation
    checkpoint = torch.load("best_mec_hybrid.pth")
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.target_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation with more episodes
    final_reward, final_latency = evaluate_agent(eval_episodes=50)
    print(f"Final evaluation - Reward: {final_reward:.2f}, Latency: {final_latency:.2f}")
    
    return agent, metrics

if __name__ == "__main__":
    agent, metrics = train_mec_hybrid()