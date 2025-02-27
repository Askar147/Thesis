import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import os
import time
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from implementations.thursday_env.mec_environment import UnifiedMECEnvironment

class EdgeConv(nn.Module):
    """Enhanced Edge Convolution layer for MEC with weighted message passing"""
    def __init__(self, in_channels, out_channels, aggr='max'):
        super(EdgeConv, self).__init__()
        
        # Edge MLP for processing node pairs
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        # Get source and target node features
        source, target = edge_index
        source_x = x[source]
        target_x = x[target]
        
        # Concatenate source and target features
        edge_features = torch.cat([source_x, target_x], dim=1)
        
        # Apply edge MLP
        edge_features = self.edge_mlp(edge_features)
        
        # Apply edge weights if available
        if edge_attr is not None:
            edge_features = edge_features * edge_attr.unsqueeze(1)
        
        # Aggregate messages to target nodes
        out = torch.zeros_like(x)
        out.index_add_(0, target, edge_features)
        
        return out


class MECGNN(nn.Module):
    """GNN model for MEC task offloading with enhanced graph representation"""
    def __init__(self, node_features, hidden_dim=64, output_dim=10, num_layers=3, 
                 dropout=0.1, use_gat=True, num_heads=2, residual=True):
        super(MECGNN, self).__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_gat = use_gat
        self.residual = residual
        
        # Node feature embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.edge_convs = nn.ModuleList()
        
        for i in range(num_layers):
            # Choose layer type
            if use_gat:
                # GAT layer with multi-head attention
                self.convs.append(GATConv(
                    hidden_dim, 
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout
                ))
            else:
                # GCN layer
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            # Edge convolution
            self.edge_convs.append(EdgeConv(hidden_dim, hidden_dim))
            
            # Layer normalization
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Task feature embedding (separate from server nodes)
        self.task_embedding = nn.Sequential(
            nn.Linear(4, hidden_dim),  # Embed task features: size, deadline, priority
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # Pooling mechanism with attention
        self.pool_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Q-value prediction heads with dueling architecture
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, edge_index, edge_attr=None, task_features=None, batch=None):
        # Embed node features
        x = self.node_embedding(x)
        x_original = x.clone()  # Store original for residual connections
        
        # Process graph with multiple GNN layers
        for i in range(self.num_layers):
            # Apply graph convolution
            x1 = self.convs[i](x, edge_index)
            
            # Apply edge convolution if edge features are provided
            if edge_attr is not None:
                x2 = self.edge_convs[i](x, edge_index, edge_attr)
                x = x1 + x2
            else:
                x = x1
            
            # Apply normalization
            x = self.norms[i](x)
            x = F.leaky_relu(x)
            
            # Add residual connection
            if self.residual and i > 0:
                x = x + x_original
                x_original = x.clone()
        
        # Global pooling with attention
        if batch is not None:
            # If batched, use proper global pooling
            node_weights = self.pool_attention(x).squeeze(-1)
            node_weights = torch.softmax(node_weights, dim=0)
            graph_features = global_mean_pool(x * node_weights.unsqueeze(-1), batch)
        else:
            # If single graph, use simple attention pooling
            node_weights = self.pool_attention(x).squeeze(-1)
            node_weights = torch.softmax(node_weights, dim=0)
            graph_features = (x * node_weights.unsqueeze(-1)).sum(dim=0)
        
        # Process task features
        if task_features is not None:
            task_embedded = self.task_embedding(task_features)
            # Combine graph and task features
            combined_features = torch.cat([graph_features, task_embedded], dim=-1)
        else:
            # If no task features, duplicate graph features
            combined_features = torch.cat([graph_features, graph_features], dim=-1)
        
        # Dueling architecture
        value = self.value_head(combined_features)
        advantage = self.advantage_head(combined_features)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values


class PrioritizedReplayBuffer:
    """Enhanced Prioritized Experience Replay buffer with better error handling"""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta_start  # Importance sampling exponent
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        
        # Define transition tuple for better organization
        self.Transition = namedtuple('Transition', 
                                     ('state', 'action', 'reward', 'next_state', 'done', 
                                      'task_features', 'edge_index', 'edge_attr'))
    
    def push(self, state, action, reward, next_state, done, task_features, edge_index, edge_attr):
        # Use max priority for new experiences
        priority = self.max_priority
        
        # Create transition
        transition = self.Transition(state, action, reward, next_state, done, 
                                    task_features, edge_index, edge_attr)
        
        # Add to buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        # Update priority
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        # Check if buffer has enough samples
        if len(self.buffer) == 0:
            return None
            
        buffer_len = min(len(self.buffer), self.capacity)
        
        # Update beta for importance sampling
        self.beta = min(1.0, self.beta + (1.0 - self.beta_start) / self.beta_frames * self.frame)
        self.frame += 1
        
        # Get priorities and convert to probabilities
        priorities = self.priorities[:buffer_len] ** self.alpha
        probs = priorities / (priorities.sum() + 1e-6)
        
        # Sample indices based on probabilities
        indices = np.random.choice(buffer_len, batch_size, p=probs)
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (buffer_len * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        weights = torch.FloatTensor(weights)
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            # Clip priority for stability
            priority = np.clip(priority, 1e-8, 100.0)
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class MECGNNAgent:
    """GNN-based agent for MEC task offloading with advanced training strategies"""
    def __init__(self, state_dim, action_dim, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # GNN model settings
        self.node_features = 4  # Features per server node (speed, load, network, distance)
        self.hidden_dim = 128
        self.use_gat = True
        self.num_heads = 4
        self.residual = True
        
        # Initialize policy and target networks
        self.policy_net = MECGNN(
            node_features=self.node_features,
            hidden_dim=self.hidden_dim,
            output_dim=action_dim,
            num_layers=3,
            dropout=0.1,
            use_gat=self.use_gat,
            num_heads=self.num_heads,
            residual=self.residual
        ).to(device)
        
        self.target_net = MECGNN(
            node_features=self.node_features,
            hidden_dim=self.hidden_dim,
            output_dim=action_dim,
            num_layers=3,
            dropout=0,  # No dropout in target network
            use_gat=self.use_gat,
            num_heads=self.num_heads,
            residual=self.residual
        ).to(device)
        
        # Copy parameters from policy to target
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network always in eval mode
        
        # Initialize optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=0.0003,
            weight_decay=1e-5,
            amsgrad=True
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=1e-5
        )
        
        # Initialize replay buffer with prioritized experience replay
        self.replay_buffer = PrioritizedReplayBuffer(100000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.01  # Soft update parameter
        self.batch_size = 64
        self.min_replay_size = 1000
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.exploration_steps = 0
        
        # N-step returns for temporal difference learning
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step)
        
        # Performance tracking
        self.training_info = {
            'losses': [],
            'q_values': [],
            'td_errors': []
        }
    
    def preprocess_state(self, state, env):
        """
        Convert environment state to graph representation for GNN.
        
        Args:
            state: State dictionary from the environment
            env: The UnifiedMECEnvironment instance
            
        Returns:
            node_features: Server node features
            task_features: Task-specific features
            edge_index: Edge connectivity
            edge_attr: Edge weights
        """
        # Extract server features
        server_speeds = state['server_speeds']
        server_loads = state['server_loads']
        network_conditions = state['network_conditions']
        server_distances = state['server_distances']
        energy_consumption = state['energy_consumption']
        
        # Extract task features
        task_size = state['task_size'][0]
        task_deadline = state['task_deadline'][0]
        task_priority = state['task_priority'][0]
        
        # Create node features for servers
        node_features = torch.FloatTensor(np.stack([
            server_speeds,
            server_loads,
            network_conditions,
            server_distances
        ], axis=1)).to(self.device)
        
        # Create task features
        task_features = torch.FloatTensor([
            task_size,
            task_deadline,
            task_priority,
            1.0  # Constant feature to indicate it's a task
        ]).unsqueeze(0).to(self.device)
        
        # Get edge information from environment
        edge_index, edge_weights = env.get_edge_data()
        edge_index = torch.LongTensor(edge_index).to(self.device)
        edge_attr = torch.FloatTensor(edge_weights).to(self.device)
        
        return node_features, task_features, edge_index, edge_attr
    
    def select_action(self, state, env, evaluate=False):
        """
        Select action using epsilon-greedy policy with server load-aware exploration
        
        Args:
            state: State dictionary from the environment
            env: The UnifiedMECEnvironment instance
            evaluate: Whether to evaluate (no exploration)
            
        Returns:
            action: Selected server index
        """
        # Extract valid server mask (servers not overloaded)
        valid_mask = env.get_valid_server_mask()
        
        # Epsilon-greedy exploration
        if not evaluate and random.random() < self.epsilon:
            # Smart exploration: bias toward less loaded servers
            if random.random() < 0.7:  # 70% of exploration uses load-awareness
                server_probs = (1.0 - np.array(state['server_loads'])) * valid_mask
                server_probs = server_probs / (server_probs.sum() + 1e-8)
                return np.random.choice(self.action_dim, p=server_probs)
            else:
                # Random exploration among valid servers
                valid_servers = np.where(valid_mask > 0)[0]
                if len(valid_servers) > 0:
                    return np.random.choice(valid_servers)
                return random.randrange(self.action_dim)
        
        # Process state for GNN
        with torch.no_grad():
            node_features, task_features, edge_index, edge_attr = self.preprocess_state(state, env)
            
            # Forward pass through the policy network
            q_values = self.policy_net(node_features, edge_index, edge_attr, task_features)
            
            # Apply mask to q_values (set invalid servers to very negative value)
            if not all(valid_mask == 1.0):
                q_values_np = q_values.cpu().numpy()
                q_values_np[np.where(valid_mask == 0)] = -1e9
                q_values = torch.FloatTensor(q_values_np).to(self.device)
            
            # Select action with highest Q-value
            return q_values.argmax().item()
    
    def _calculate_n_step_returns(self, n_step_buffer):
        """Calculate n-step returns for more accurate value estimates"""
        reward, next_state, done = n_step_buffer[-1][2:5]
        
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, next_s, d = transition[2:5]
            
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (next_s, d) if d else (next_state, done)
            
        return reward, next_state, done
    
    def store_transition(self, state, action, reward, next_state, done, env):

        node_features, task_features, edge_index, edge_attr = self.preprocess_state(state, env)
        
        # Add transition to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done, 
                                task_features, edge_index, edge_attr))
        
        # Once we have enough transitions for n-step return
        if len(self.n_step_buffer) == self.n_step:
            # Get initial state and action
            init_state, init_action = self.n_step_buffer[0][:2]
            
            # Calculate n-step return
            n_step_reward, n_step_next_state, n_step_done = self._calculate_n_step_returns(self.n_step_buffer)
            
            # Get task features and graph structure from the first transition
            init_task_features, init_edge_index, init_edge_attr = self.n_step_buffer[0][5:]
            
            # Add to replay buffer
            self.replay_buffer.push(init_state, init_action, n_step_reward, n_step_next_state, n_step_done,
                                init_task_features, init_edge_index, init_edge_attr)
        
        # If episode ends before n-step buffer is full
        if done and len(self.n_step_buffer) > 0:
            # Get first state and action
            init_state, init_action = self.n_step_buffer[0][:2]
            
            # Calculate multi-step return
            n_step_reward, n_step_next_state, n_step_done = self._calculate_n_step_returns(self.n_step_buffer)
            
            # Get task features and graph structure from the first transition
            init_task_features, init_edge_index, init_edge_attr = self.n_step_buffer[0][5:]
            
            # Add to replay buffer
            self.replay_buffer.push(init_state, init_action, n_step_reward, n_step_next_state, n_step_done,
                                init_task_features, init_edge_index, init_edge_attr)
            
            # Clear n-step buffer
            self.n_step_buffer.clear()
    
        # For immediate learning, also store the current transition in the replay buffer
        # This is ADDED to ensure we're storing transitions even when n-step buffer isn't full
        self.replay_buffer.push(state, action, reward, next_state, done,
                            task_features, edge_index, edge_attr)
        
    def optimize_model(self):
        """Perform one step of optimization on the model"""
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        # Sample from replay buffer
        transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
        if transitions is None:
            return None
        
        # Process batch
        batch_states = [t.state for t in transitions]
        batch_actions = torch.tensor([t.action for t in transitions], device=self.device)
        batch_rewards = torch.tensor([t.reward for t in transitions], device=self.device, dtype=torch.float)
        batch_next_states = [t.next_state for t in transitions]
        batch_dones = torch.tensor([t.done for t in transitions], device=self.device, dtype=torch.float)
        
        # Process features for GNN
        batch_task_features = torch.cat([t.task_features for t in transitions])
        
        # Handle edge indices and attributes
        # This is complex since edge indices need to be shifted for batching
        batch_edge_indices = []
        batch_edge_attrs = []
        current_shift = 0
        num_servers = self.action_dim
        
        for t in transitions:
            # Copy edge index and shift node indices for proper batching
            edge_index = t.edge_index.clone()
            edge_index[0] += current_shift
            edge_index[1] += current_shift
            batch_edge_indices.append(edge_index)
            
            # Add edge attributes
            batch_edge_attrs.append(t.edge_attr)
            
            # Update shift for next graph
            current_shift += num_servers
        
        # Concatenate edge information
        batch_edge_index = torch.cat(batch_edge_indices, dim=1)
        batch_edge_attr = torch.cat(batch_edge_attrs)
        
        # Process node features from states
        batch_node_features = []
        for state in batch_states:
            # Extract server features
            node_features = torch.FloatTensor(np.stack([
                state['server_speeds'],
                state['server_loads'],
                state['network_conditions'],
                state['server_distances']
            ], axis=1)).to(self.device)
            batch_node_features.append(node_features)
        
        batch_next_node_features = []
        for state in batch_next_states:
            # Extract server features for next states
            node_features = torch.FloatTensor(np.stack([
                state['server_speeds'],
                state['server_loads'],
                state['network_conditions'],
                state['server_distances']
            ], axis=1)).to(self.device)
            batch_next_node_features.append(node_features)
        
        # Concatenate all node features
        batch_node_features = torch.cat(batch_node_features)
        batch_next_node_features = torch.cat(batch_next_node_features)
        
        # Create batch index tensor for global pooling
        batch_idx = torch.repeat_interleave(
            torch.arange(self.batch_size, device=self.device),
            repeats=torch.tensor([num_servers] * self.batch_size, device=self.device)
        )
        
        # Create next state task features
        batch_next_task_features = torch.stack([
            torch.FloatTensor([
                state['task_size'][0],
                state['task_deadline'][0],
                state['task_priority'][0],
                1.0  # Constant feature
            ]) for state in batch_next_states
        ]).to(self.device)
        
        # Import the importance sampling weights
        weights = weights.to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(
            batch_node_features, 
            batch_edge_index, 
            batch_edge_attr, 
            batch_task_features,
            batch_idx
        )
        current_q = current_q_values.gather(1, batch_actions.unsqueeze(1))
        
        # Compute next Q values with Double DQN approach
        with torch.no_grad():
            # Get actions from policy network
            next_q_policy = self.policy_net(
                batch_next_node_features, 
                batch_edge_index, 
                batch_edge_attr, 
                batch_next_task_features,
                batch_idx
            )
            next_actions = next_q_policy.max(1)[1].unsqueeze(1)
            
            # Get Q-values from target network for those actions
            next_q_target = self.target_net(
                batch_next_node_features, 
                batch_edge_index, 
                batch_edge_attr, 
                batch_next_task_features,
                batch_idx
            )
            next_q = next_q_target.gather(1, next_actions)
            
            # Compute expected Q values
            expected_q = batch_rewards.unsqueeze(1) + \
                        (1 - batch_dones.unsqueeze(1)) * self.gamma * next_q
        
        # Compute loss with prioritized experience replay
        td_errors = torch.abs(current_q - expected_q).detach().cpu().numpy()
        loss = (weights.unsqueeze(1) * F.smooth_l1_loss(current_q, expected_q, reduction='none')).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update replay buffer priorities
        self.replay_buffer.update_priorities(indices, td_errors.squeeze() + 1e-6)
        
        # Update target network with soft update
        self._soft_update_target()
        
        # Track metrics
        self.training_info['losses'].append(loss.item())
        self.training_info['q_values'].append(current_q.mean().item())
        self.training_info['td_errors'].append(td_errors.mean())
        
        return loss.item()
    
    def _soft_update_target(self):
        """Soft update target network parameters"""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )
    
    def update_exploration(self):
        """Update exploration rate with adaptive decay"""
        self.exploration_steps += 1
        
        # Adaptive epsilon decay - slower at start, faster in the middle, then very slow at end
        if self.exploration_steps < 1000:
            # Slow decay at start
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.999)
        elif self.exploration_steps < 10000:
            # Faster decay in the middle
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.995)
        else:
            # Very slow decay at end
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.999)
    
    def save_model(self, path):
        """Save model weights and training information"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_info': self.training_info,
            'epsilon': self.epsilon,
            'exploration_steps': self.exploration_steps
        }, path)
        
    def load_model(self, path):
        """Load model weights and training information"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_info = checkpoint['training_info']
        self.epsilon = checkpoint['epsilon']
        self.exploration_steps = checkpoint['exploration_steps']


def train_mec_gnn(env, agent, num_episodes=1000, max_steps=100, 
                 eval_frequency=25, save_frequency=100, checkpoint_dir='models'):
    """
    Train the GNN agent in the MEC environment
    
    Args:
        env: UnifiedMECEnvironment instance
        agent: MECGNNAgent instance
        num_episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        eval_frequency: Frequency of evaluation
        save_frequency: Frequency of saving checkpoints
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        agent: Trained agent
        metrics: Training metrics
    """
    # Create directory for checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize metrics
    metrics = {
        'episode_rewards': [],
        'eval_rewards': [],
        'episode_latencies': [],
        'eval_latencies': [],
        'episode_losses': [],
        'q_values': [],
        'epsilon_values': [],
        'server_selection': []
    }
    
    # Best model tracking
    best_eval_reward = float('-inf')
    no_improvement_count = 0
    early_stop_patience = 20
    
    # Start timer
    start_time = time.time()
    
    # Evaluation function
    def evaluate_agent(eval_episodes=10):
        eval_rewards = []
        eval_latencies = []
        server_selections = []
        
        # Store current epsilon and set to minimum for evaluation
        current_epsilon = agent.epsilon
        agent.epsilon = agent.epsilon_min / 2  # Even less exploration during eval
        
        for _ in range(eval_episodes):
            state = env.reset()
            episode_reward = 0
            episode_latencies = []
            episode_servers = []
            
            # Clear n-step buffer
            agent.n_step_buffer.clear()
            
            for step in range(max_steps):
                action = agent.select_action(state, env, evaluate=True)
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
        
    # Main training loop
    for episode in range(num_episodes):
        # Reset environment and metrics
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        episode_latencies = []
        episode_servers = []
        
        # Clear n-step buffer at start of episode
        agent.n_step_buffer.clear()
        
        for step in range(max_steps):
            # Select action and take step
            action = agent.select_action(state, env)
            next_state, reward, done, info = env.step(action)
            
            # Store transition and train
            agent.store_transition(state, action, reward, next_state, done, env)
            loss = agent.optimize_model()
            
            if loss is not None:
                episode_losses.append(loss)
            
            episode_latencies.append(info['total_latency'])
            episode_servers.append(action)
            state = next_state
            episode_reward += reward
        
        # Update exploration rate
        agent.update_exploration()
        
        # Update metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_latencies'].append(np.mean(episode_latencies))
        metrics['epsilon_values'].append(agent.epsilon)
        
        # Track server selection distribution
        server_counts = np.bincount(episode_servers, minlength=agent.action_dim)
        server_distribution = server_counts / len(episode_servers)
        metrics['server_selection'].append(server_distribution)
        
        if episode_losses:
            metrics['episode_losses'].append(np.mean(episode_losses))
            
        if len(agent.training_info['q_values']) > 0:
            metrics['q_values'].append(np.mean(agent.training_info['q_values'][-100:]))
        
        # Step the scheduler every few episodes
        if episode % 5 == 0:
            agent.scheduler.step()
        
        # Evaluation
        if episode % eval_frequency == 0 or episode == num_episodes - 1:
            # Calculate time elapsed
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            # Run evaluation
            eval_reward, eval_latency, server_diversity = evaluate_agent()
            metrics['eval_rewards'].append(eval_reward)
            metrics['eval_latencies'].append(eval_latency)
            
            # Print progress
            print(f"Episode {episode}/{num_episodes} [{time_str}], "
                  f"Train Reward: {episode_reward:.2f}, "
                  f"Eval Reward: {eval_reward:.2f}, "
                  f"Latency: {eval_latency:.4f}, "
                  f"Epsilon: {agent.epsilon:.4f}, "
                  f"Server Diversity: {server_diversity:.1f}/{agent.action_dim}")
            
            # Save best model
            if eval_reward > best_eval_reward:
                improvement = eval_reward - best_eval_reward
                best_eval_reward = eval_reward
                no_improvement_count = 0
                
                # Save best model
                agent.save_model(f"{checkpoint_dir}/best_gnn_model.pt")
                print(f"New best model saved with reward: {best_eval_reward:.2f} (improvement: {improvement:.2f})")
            else:
                no_improvement_count += 1
                print(f"No improvement for {no_improvement_count} evaluations")
            
            # Early stopping
            if no_improvement_count >= early_stop_patience:
                print(f"Early stopping triggered after {episode} episodes")
                break
        
        # Save checkpoint periodically
        if episode % save_frequency == 0 and episode > 0:
            agent.save_model(f"{checkpoint_dir}/gnn_checkpoint_ep{episode}.pt")
    
    # Final evaluation with best model
    print("Training completed. Loading best model for final evaluation...")
    agent.load_model(f"{checkpoint_dir}/best_gnn_model.pt")
    final_reward, final_latency, final_diversity = evaluate_agent(eval_episodes=30)
    print(f"Final evaluation - Reward: {final_reward:.2f}, "
          f"Latency: {final_latency:.4f}, "
          f"Server Diversity: {final_diversity:.1f}/{agent.action_dim}")
    
    return agent, metrics


def plot_training_metrics(metrics, filename='gnn_training_metrics.png'):
    """Plot and save training metrics"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(20, 15))
    
    # Plot rewards
    plt.subplot(3, 2, 1)
    plt.plot(metrics['episode_rewards'], alpha=0.3, label='Episode Rewards')
    plt.plot(np.convolve(metrics['episode_rewards'], np.ones(20)/20, mode='valid'), 
             label='Moving Avg (20 episodes)', linewidth=2)
    
    eval_indices = [i * eval_frequency for i in range(len(metrics['eval_rewards']))]
    plt.plot(eval_indices, metrics['eval_rewards'], 'r-o', label='Evaluation Rewards', linewidth=2)
    
    plt.title('Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot latencies
    plt.subplot(3, 2, 2)
    plt.plot(metrics['episode_latencies'], alpha=0.3, label='Episode Latencies')
    plt.plot(np.convolve(metrics['episode_latencies'], np.ones(20)/20, mode='valid'),
             label='Moving Avg (20 episodes)', linewidth=2)
    
    plt.plot(eval_indices, metrics['eval_latencies'], 'r-o', label='Evaluation Latencies', linewidth=2)
    
    plt.title('Latencies')
    plt.xlabel('Episode')
    plt.ylabel('Latency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot losses
    plt.subplot(3, 2, 3)
    if metrics.get('episode_losses'):
        plt.plot(metrics['episode_losses'], alpha=0.5)
        plt.title('Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
    
    # Plot Q-values
    plt.subplot(3, 2, 4)
    if metrics.get('q_values'):
        plt.plot(metrics['q_values'], alpha=0.5)
        plt.title('Average Q-Values')
        plt.xlabel('Episode')
        plt.ylabel('Q-Value')
        plt.grid(True, alpha=0.3)
    
    # Plot epsilon
    plt.subplot(3, 2, 5)
    plt.plot(metrics['epsilon_values'])
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True, alpha=0.3)
    
    # Plot server selection distribution (heatmap of last few episodes)
    plt.subplot(3, 2, 6)
    if len(metrics['server_selection']) > 0:
        num_servers = len(metrics['server_selection'][0])
        recent_selections = metrics['server_selection'][-min(20, len(metrics['server_selection'])):]
        
        plt.imshow(recent_selections, aspect='auto', cmap='viridis')
        plt.colorbar(label='Selection Frequency')
        plt.title('Recent Server Selection Distribution')
        plt.xlabel('Server ID')
        plt.ylabel('Recent Episodes')
        plt.xticks(range(num_servers))
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    # Create the UnifiedMECEnvironment
    env = UnifiedMECEnvironment(
        num_edge_servers=10,
        continuous_action=False,
        history_length=20,
        difficulty='normal',
        seed=42
    )
    
    # Calculate state dimension (flattened state)
    state = env.reset()
    state_dim = len(env.flatten_state(state))
    action_dim = env.num_edge_servers
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create the GNN agent
    agent = MECGNNAgent(
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    # Create checkpoint directory
    checkpoint_dir = "mec_gnn_models"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train the agent
    trained_agent, metrics = train_mec_gnn(
        env=env,
        agent=agent,
        num_episodes=1000,
        max_steps=100,
        eval_frequency=25,
        save_frequency=100,
        checkpoint_dir=checkpoint_dir
    )
    
    # Plot training metrics
    plot_training_metrics(metrics, filename=f"{checkpoint_dir}/training_metrics.png")
    
    print("Training and evaluation completed. Check the metrics plot.")
    
    # Optional: Test the trained agent
    print("\nRunning final test episodes with trained agent...")
    
    for episode in range(5):
        state = env.reset()
        episode_reward = 0
        selected_servers = []
        
        for step in range(100):
            # Select action
            action = trained_agent.select_action(state, env, evaluate=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Print step details
            if step % 20 == 0:  # Print every 20 steps to avoid too much output
                print(f"Episode {episode}, Step {step}, Selected Server: {action}, "
                      f"Reward: {reward:.2f}, Latency: {info['total_latency']:.4f}")
            
            selected_servers.append(action)
            state = next_state
            episode_reward += reward
        
        # Calculate server diversity
        server_diversity = len(set(selected_servers))
        
        print(f"Episode {episode} - Total Reward: {episode_reward:.2f}, "
              f"Server Diversity: {server_diversity}/{env.num_edge_servers}")
        print(f"Server selection distribution: {np.bincount(selected_servers, minlength=env.num_edge_servers)}")
        print("-" * 50)