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
    """Transformer for temporal patterns in MEC"""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_length, input_dim]
        batch_size = x.size(0)
        
        # Project input to d_model dimension
        x = self.input_projection(x)  # [batch_size, seq_length, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)  # [batch_size, seq_length, d_model]
        
        # Apply transformer
        output = self.transformer(x, mask)  # [batch_size, seq_length, d_model]
        
        # Average over sequence length
        output = output.mean(dim=1)  # [batch_size, d_model]
        
        return output

class GNNComponent(nn.Module):
    """GNN for server relationships"""
    def __init__(self, node_features, hidden_dim=64):
        super().__init__()
        
        self.conv1 = GATConv(node_features, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index):
        x = self.ln1(F.relu(self.conv1(x, edge_index)))
        x = self.ln2(F.relu(self.conv2(x, edge_index)))
        return x

class VAEComponent(nn.Module):
    """VAE for state distributions"""
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class SACComponent(nn.Module):
    """SAC for decision making"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # Q networks
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        # state shape: [batch_size, state_dim]
        policy_features = self.policy(state)
        mean = self.mean_head(policy_features)
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
        
        # Initialize components
        self.transformer = TransformerComponent(state_dim, hidden_dim)
        self.gnn = GNNComponent(4)  # 4 features per server
        self.vae = VAEComponent(state_dim, hidden_dim, latent_dim)
                # Update SAC: change input dimension from hidden_dim * 3 to hidden_dim.
        self.sac = SACComponent(hidden_dim, action_dim)
        
        # Add projection layers for spatial (GNN) and VAE outputs:
        self.spatial_proj = nn.Linear(64, hidden_dim)      # GNN output is 64-dim, project to 128.
        self.vae_proj = nn.Linear(state_dim, hidden_dim)     # VAE output is state_dim (41), project to 128.
        
        # Fusion network remains the same (combining three 128-dim vectors -> 384):
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # 384 -> 256
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim)         # 256 -> 128
        )
    
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
        
        return node_features, state_tensor, batch_size
        
    def forward(self, state_sequence, current_states):
        # Get batch size and ensure proper dimensions
        batch_size = state_sequence.size(0)
        
        # Process temporal patterns with transformer
        temporal_features = self.transformer(state_sequence)  # [batch_size, hidden_dim] (128)
        
        # Process current states
        node_features, state_tensor, _ = self._process_batch_states(current_states)
        
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
            gnn_output = self.gnn(batch_nodes, edge_index)  # [num_servers, 64]
            spatial_features_list.append(gnn_output.mean(0))  # [64]
        
        spatial_features = torch.stack(spatial_features_list)  # [batch_size, 64]
        # Project spatial features from 64-dim to 128-dim:
        spatial_features = self.spatial_proj(spatial_features)  # [batch_size, hidden_dim]
        
        # Process state distribution with VAE
        vae_output, mu, logvar = self.vae(state_tensor)  # vae_output: [batch_size, state_dim] (e.g., 41)
        # Project VAE output to 128-dim:
        vae_output = self.vae_proj(vae_output)  # [batch_size, hidden_dim]
        
        # Ensure all features have the same batch size
        assert temporal_features.size(0) == batch_size
        assert spatial_features.size(0) == batch_size
        assert vae_output.size(0) == batch_size
        
        # Combine features (now 128 + 128 + 128 = 384 dimensions)
        combined_features = torch.cat([
            temporal_features,
            spatial_features,
            vae_output
        ], dim=1)  # [batch_size, hidden_dim * 3]
        
        # Fuse features
        fused_features = self.fusion(combined_features)  # [batch_size, hidden_dim] (128)
        
        # Get action distribution and Q-values from SAC
        mean, log_std, q1, q2 = self.sac(fused_features)
        
        return {
            'policy': (mean, log_std),
            'q_values': (q1, q2),
            'vae_params': (mu, logvar),
            'fused_features': fused_features
        }

    

class HybridReplayBuffer:
    """Replay buffer for hybrid system"""
    def __init__(self, capacity, seq_length):
        self.buffer = deque(maxlen=capacity)
        self.seq_length = seq_length
        
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
        
        self.buffer.append((state_copy, action, reward, next_state_copy, done))
    
    def sample(self, batch_size):
        # Ensure enough samples for sequences
        valid_idx = len(self.buffer) - self.seq_length
        if valid_idx < batch_size:
            return None
        
        # Sample starting indices
        start_indices = np.random.randint(0, valid_idx, size=batch_size)
        
        batch_state_seqs = []
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        for start_idx in start_indices:
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
                batch_rewards, batch_next_states, batch_dones)
    
    def __len__(self):
        return len(self.buffer)
    

class MECHybridAgent:
    """Hybrid agent combining Transformer, GNN, VAE, and SAC for MEC task offloading"""
    def __init__(self, state_dim, action_dim, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize hybrid system
        self.model = MECHybridSystem(state_dim, action_dim).to(device)
        self.target_model = MECHybridSystem(state_dim, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Initialize optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        
        # Initialize alpha (SAC temperature parameter)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        
        # Initialize replay buffer
        self.replay_buffer = HybridReplayBuffer(100000, seq_length=10)
        
        # Initialize episode buffer for sequence building
        self.episode_buffer = []
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.min_replay_size = 1000
        self.target_entropy = -action_dim
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
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
    
    def select_action(self, state, evaluate=False):
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
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            outputs = self.model(state_sequence_tensor, state)
            mean, log_std = outputs['policy']
            
            if evaluate:
                return torch.argmax(mean).item()
            
            std = log_std.exp()
            dist = Normal(mean, std)
            action = dist.rsample()
            return torch.argmax(action).item()
    
    def train(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        # Sample from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None
        
        state_seqs, states, actions, rewards, next_states, dones = batch
        
        # Prepare tensors
        state_seqs_tensor = torch.stack([
            self._prepare_sequence(seq) for seq in state_seqs
        ]).to(self.device)
        
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Get current model outputs
        outputs = self.model(state_seqs_tensor, states)
  # Use first state as current
        mean, log_std = outputs['policy']
        q1, q2 = outputs['q_values']
        mu, logvar = outputs['vae_params']
        
        # Get target model outputs for next states
        with torch.no_grad():
            next_outputs = self.target_model(state_seqs_tensor, next_states)  # Use first next state
            next_q1, next_q2 = next_outputs['q_values']
            next_value = torch.min(next_q1, next_q2)
            target_q = rewards_tensor.unsqueeze(1) + \
                      (1 - dones_tensor.unsqueeze(1)) * self.gamma * next_value
        
        # Compute losses
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        std = log_std.exp()
        dist = Normal(mean, std)
        action_probs = dist.rsample()
        log_probs = dist.log_prob(action_probs)
        
        alpha = self.log_alpha.exp()
        policy_loss = (alpha * log_probs - torch.min(q1, q2)).mean()
        
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        vae_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = q_loss + policy_loss + 0.1 * vae_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update target network
        self._soft_update_target_network()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'vae_loss': vae_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item()
        }
    
    def _soft_update_target_network(self):
        """Soft update target network parameters"""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

def train_mec_hybrid():
    """Training loop for hybrid system"""
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
    num_episodes = 1000
    max_steps = 100
    eval_frequency = 50
    
    # Metrics tracking
    metrics = {
        'rewards': [],
        'q_losses': [],
        'policy_losses': [],
        'vae_losses': [],
        'alpha_losses': [],
        'alphas': [],
        'latencies': [],
        'server_loads': [],
        'avg_rewards': []
    }
    
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
            losses = agent.train()
            
            if losses is not None:
                episode_losses.append(losses)
            
            episode_latencies.append(info['total_latency'])
            state = next_state
            episode_reward += reward
        
        # Update metrics
        metrics['rewards'].append(episode_reward)
        metrics['latencies'].append(np.mean(episode_latencies))
        metrics['server_loads'].append(np.mean(state['server_loads']))
        
        if episode_losses:
            avg_losses = {k: np.mean([loss[k] for loss in episode_losses]) 
                        for k in episode_losses[0].keys()}
            metrics['q_losses'].append(avg_losses['q_loss'])
            metrics['policy_losses'].append(avg_losses['policy_loss'])
            metrics['vae_losses'].append(avg_losses['vae_loss'])
            metrics['alpha_losses'].append(avg_losses['alpha_loss'])
            metrics['alphas'].append(avg_losses['alpha'])
        
        # Calculate running average
        if episode >= eval_frequency:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            metrics['avg_rewards'].append(avg_reward)
        
        # Print progress
        if episode % eval_frequency == 0:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            avg_latency = np.mean(metrics['latencies'][-eval_frequency:])
            print(f"Episode {episode}/{num_episodes}, "
                  f"Average Reward: {avg_reward:.2f}, "
                  f"Average Latency: {avg_latency:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return agent, metrics

if __name__ == "__main__":
    agent, metrics = train_mec_hybrid()