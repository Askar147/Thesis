import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data, Batch
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from DQN_second import MECEnvironment


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
    """Improved Graph Neural Network for MEC task offloading with multi-head attention and skip connections"""
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
            nn.Linear(hidden_dim, num_servers)
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
    
    def create_graph_from_state(self, state_dict):
        """Convert MEC environment state to graph with improved feature engineering"""
        # Extract state components and convert to tensors on the correct device
        task_size = torch.FloatTensor(state_dict['task_size']).to(self.device)
        server_speeds = torch.FloatTensor(state_dict['server_speeds']).to(self.device)
        server_loads = torch.FloatTensor(state_dict['server_loads']).to(self.device)
        network_conditions = torch.FloatTensor(state_dict['network_conditions']).to(self.device)
        server_distances = torch.FloatTensor(state_dict['server_distances']).to(self.device)
        
        # Compute efficiency metrics (feature engineering)
        processing_efficiency = server_speeds * (1 - server_loads)
        
        # Create task node features
        padding = torch.zeros(3, device=self.device)
        task_features = torch.cat([task_size, padding]).unsqueeze(0)  # Shape: [1, 4]
        
        # Create server node features with additional derived metrics
        server_features = []
        for i in range(self.num_servers):
            server_feature = torch.cat([
                server_speeds[i].unsqueeze(0),
                server_loads[i].unsqueeze(0),
                network_conditions[i].unsqueeze(0),
                server_distances[i].unsqueeze(0)
            ]).unsqueeze(0)  # Shape: [1, 4]
            server_features.append(server_feature)
        
        server_features = torch.cat(server_features, dim=0)  # Shape: [num_servers, 4]
        
        # Combine task and server features
        node_features = torch.cat([task_features, server_features], dim=0)  # Shape: [num_servers + 1, 4]
        
        # Create edges (fully connected between task and servers)
        edge_index = []
        edge_attr = []
        
        # Add edges from task to each server with richer edge features
        for i in range(self.num_servers):
            # Task -> Server edge features
            task_to_server = [
                network_conditions[i].item(),
                server_distances[i].item(),
                processing_efficiency[i].item()  # Add derived feature
            ]
            
            # Task -> Server
            edge_index.append([0, i + 1])
            edge_attr.append(task_to_server)
            
            # Server -> Task (same features, different direction)
            edge_index.append([i + 1, 0])
            edge_attr.append(task_to_server)
        
        # Add server-to-server edges for modeling server cooperation/competition
        for i in range(self.num_servers):
            for j in range(i+1, self.num_servers):
                # Calculate distance between servers (example metric)
                server_to_server_distance = abs(server_distances[i].item() - server_distances[j].item())
                
                # Server i -> Server j
                edge_index.append([i + 1, j + 1])
                edge_attr.append([
                    (network_conditions[i].item() + network_conditions[j].item()) / 2,
                    server_to_server_distance,
                    (processing_efficiency[i].item() + processing_efficiency[j].item()) / 2
                ])
                
                # Server j -> Server i (symmetric)
                edge_index.append([j + 1, i + 1])
                edge_attr.append([
                    (network_conditions[i].item() + network_conditions[j].item()) / 2,
                    server_to_server_distance,
                    (processing_efficiency[i].item() + processing_efficiency[j].item()) / 2
                ])
        
        # Convert to tensors on the correct device
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=self.device)
        
        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        return data
    
    def forward(self, state_dict):
        # Create graph from state
        data = self.create_graph_from_state(state_dict)
        
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
        
        # Get task node representation (first node)
        task_repr = x[0]
        
        # Dueling architecture: split into advantage and value
        advantage = self.advantage(task_repr)
        value = self.value(task_repr)
        
        # Combine value and advantage for Q-values using dueling technique
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

class MECGNNImprovedAgent:
    """Improved GNN-based agent for MEC task offloading with PER and target network stabilization"""
    def __init__(self, num_servers, node_features=4, edge_features=3,
                 hidden_dim=128, num_heads=4, num_layers=4,
                 lr=0.0001, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.num_servers = num_servers
        
        # Initialize networks with improved architecture
        self.policy_net = MECGraphNet(
            node_features=node_features,
            edge_features=edge_features,
            num_servers=num_servers,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=num_heads,
            device=device
        )
        
        self.target_net = MECGraphNet(
            node_features=node_features,
            edge_features=edge_features,
            num_servers=num_servers,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=num_heads,
            device=device
        )
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Freeze target network parameters
        for param in self.target_net.parameters():
            param.requires_grad = False
        
        # Initialize optimizer with weight decay (L2 regularization)
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), 
            lr=lr,
            weight_decay=1e-5,
            amsgrad=True
        )
        
        # Learning rate scheduler for adaptive learning
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
        self.gamma = 0.99
        self.tau = 0.01  # Increased for more stable target updates
        self.batch_size = 128  # Increased batch size
        self.min_replay_size = 2000  # Increased to ensure better initial learning
        self.epsilon = 1.0
        self.epsilon_min = 0.01  # Lower to ensure more exploitation
        self.epsilon_decay = 0.995
        self.update_frequency = 4  # Update policy every 4 steps
        self.steps = 0
        
        # Multi-step returns for temporal difference learning
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step)
        
        # Per-step losses for monitoring
        self.recent_losses = deque(maxlen=100)
    
    def _get_n_step_info(self):
        """Calculate n-step returns for more effective bootstrapping"""
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
            
        return reward, next_state, done
    
    def select_action(self, state, evaluate=False):
        """Select action using epsilon-greedy with adaptive temperature"""
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.num_servers)
        
        with torch.no_grad():
            q_values, _ = self.policy_net(state)
            
            # In evaluation mode, use deterministic policy
            if evaluate:
                return q_values.argmax().item()
            
            # Use softmax with temperature for better exploration
            temperature = max(0.5, 1.0 - self.steps / 10000)  # Adaptive temperature
            probs = F.softmax(q_values / temperature, dim=0)
            
            # Sometimes sample from softmax distribution instead of taking argmax
            if random.random() < 0.05:  # 5% chance for softmax sampling
                return torch.multinomial(probs, 1).item()
            else:
                return q_values.argmax().item()
    
    def train(self):
        """Train the agent with improved techniques"""
        # Only train if we have enough samples
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        self.steps += 1
        
        # Only update every few steps
        if self.steps % self.update_frequency != 0:
            return None
        
        # Sample from replay buffer with priorities
        transitions, weights, indices = self.replay_buffer.sample(self.batch_size)
        if not transitions:  # Empty buffer case
            return None
            
        batch = list(zip(*transitions))
        
        # Process batch
        state_batch = batch[0]  # List of state dictionaries
        action_batch = torch.tensor(batch[1], device=self.device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=self.device)
        next_state_batch = batch[3]  # List of state dictionaries
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
            for next_state in next_state_batch:
                # Get actions from policy network
                next_q_values, _ = self.policy_net(next_state)
                best_actions = next_q_values.argmax(dim=0).unsqueeze(0)
                
                # Get Q-values from target network
                target_q_values, target_value = self.target_net(next_state)
                
                # Use a mixture of DDQN and value estimation
                next_q = 0.8 * target_q_values.gather(0, best_actions) + 0.2 * target_value
                target_q.append(next_q)
                
            target_q = torch.cat(target_q).to(self.device)
            target_q = reward_batch.unsqueeze(1) + \
                      (1.0 - done_batch.unsqueeze(1)) * self.gamma * target_q
        
        # Compute loss with priorities
        errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

        # Update replay buffer priorities - extract the individual error values properly
        priority_updates = [float(error) + 1e-6 for error in errors.flatten()]
        self.replay_buffer.update_priorities(indices, priority_updates)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        # Update replay buffer priorities
        self.replay_buffer.update_priorities(indices, priority_updates)
  # Add small constant for stability
        
        # Soft update target network
        for target_param, policy_param in zip(self.target_net.parameters(),
                                            self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )
        
        # Update epsilon with cosine annealing
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_min + (self.epsilon - self.epsilon_min) * 
            np.cos(0.5 * np.pi * self.steps / 100000)
        )
        
        return loss.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition with n-step returns"""
        # Store in n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If n-step buffer is full, add to replay buffer with n-step returns
        if len(self.n_step_buffer) == self.n_step:
            # Get first state and action
            state, action = self.n_step_buffer[0][:2]
            
            # Get reward, next_state, and done flag from n-step return
            reward, next_state, done = self._get_n_step_info()
            
            # Store in replay buffer
            self.replay_buffer.push(state, action, reward, next_state, done)
        
        # If episode ends before n-step buffer is full
        if done:
            # Get the first state and action
            if len(self.n_step_buffer) > 0:
                state, action = self.n_step_buffer[0][:2]
                
                # Calculate n-step returns
                reward, next_state, done = self._get_n_step_info()
                
                # Store in replay buffer
                self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Clear n-step buffer at end of episode
            self.n_step_buffer.clear()
    
    def save_model(self, path):
        """Save model weights"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load_model(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']

def train_improved_mec_gnn(render=False):
    """Enhanced training loop for GNN in MEC environment with evaluation"""
    # Initialize environment
    env = MECEnvironment(num_edge_servers=10)
    
    # Initialize agent with improved architecture
    agent = MECGNNImprovedAgent(
        num_servers=env.num_edge_servers,
        node_features=4,
        edge_features=3,
        hidden_dim=128,
        num_heads=4,
        num_layers=4,
        lr=0.0001
    )
    
    # Training parameters
    num_episodes = 1000
    max_steps = 100
    eval_frequency = 25
    save_frequency = 100
    
    # Metrics tracking with expanded metrics
    metrics = {
        'rewards': [],
        'losses': [],
        'latencies': [],
        'server_loads': [],
        'avg_rewards': [],
        'eval_rewards': [],
        'selected_servers': [],
        'epsilon_values': []
    }
    
    best_eval_reward = float('-inf')
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        episode_latencies = []
        episode_servers = []
        
        # Clear n-step buffer at start of episode
        agent.n_step_buffer.clear()
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition and train
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train()
            
            if loss is not None:
                episode_losses.append(loss)
            
            episode_latencies.append(info['total_latency'])
            episode_servers.append(action)
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update metrics
        metrics['rewards'].append(episode_reward)
        metrics['latencies'].append(np.mean(episode_latencies))
        metrics['server_loads'].append(np.mean(state['server_loads']))
        metrics['selected_servers'].append(np.bincount(episode_servers, minlength=env.num_edge_servers) / len(episode_servers))
        metrics['epsilon_values'].append(agent.epsilon)
        
        if episode_losses:
            metrics['losses'].append(np.mean(episode_losses))
        
        # Calculate running average
        if episode >= 20:
            avg_reward = np.mean(metrics['rewards'][-20:])
            metrics['avg_rewards'].append(avg_reward)
            
            # Update learning rate scheduler
            agent.scheduler.step(avg_reward)
        
        # Evaluation
        if episode % eval_frequency == 0 or episode == num_episodes - 1:
            eval_reward = evaluate_agent(agent, env, 10)
            metrics['eval_rewards'].append(eval_reward)
            
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:]) if episode >= eval_frequency else 0
            avg_latency = np.mean(metrics['latencies'][-eval_frequency:]) if episode >= eval_frequency else 0
            
            print(f"Episode {episode}/{num_episodes}, "
                  f"Train Reward: {avg_reward:.2f}, "
                  f"Eval Reward: {eval_reward:.2f}, "
                  f"Average Latency: {avg_latency:.4f}, "
                  f"Epsilon: {agent.epsilon:.4f}")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save_model('best_mec_gnn_model.pt')
                print(f"New best model saved with eval reward: {eval_reward:.2f}")
        
        # Save periodically
        if episode % save_frequency == 0 and episode > 0:
            agent.save_model(f'mec_gnn_model_episode_{episode}.pt')
            
            # Plot and save metrics
            if render:
                plot_metrics(metrics, episode)
    
    # Final save
    agent.save_model('final_mec_gnn_model.pt')
    
    # Final plots
    if render:
        plot_metrics(metrics, num_episodes)
    
    return agent, metrics

def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate agent performance without exploration"""
    total_rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        max_steps = 100
        
        for step in range(max_steps):
            # Select action deterministically
            action = agent.select_action(state, evaluate=True)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Update
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)

def plot_metrics(metrics, episode):
    """Plot and save training metrics"""
    plt.figure(figsize=(20, 15))
    
    # Plot rewards
    plt.subplot(3, 2, 1)
    plt.plot(metrics['rewards'], alpha=0.3, label='Episode Rewards')
    if metrics['avg_rewards']:
        plt.plot(range(20, len(metrics['rewards'])), metrics['avg_rewards'], label='Average Rewards (20 ep)', linewidth=2)
    if metrics['eval_rewards']:
        plt.plot(range(0, len(metrics['rewards']), 25)[:len(metrics['eval_rewards'])], 
                 metrics['eval_rewards'], 'r-', label='Evaluation Rewards', linewidth=2)
    plt.title('Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot losses
    plt.subplot(3, 2, 2)
    if metrics['losses']:
        plt.plot(metrics['losses'])
        plt.title('Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
    # Plot latencies
    plt.subplot(3, 2, 3)
    plt.plot(metrics['latencies'])
    plt.title('Average Latency')
    plt.xlabel('Episode')
    plt.ylabel('Latency')
    plt.grid(True, alpha=0.3)
    
    # Plot server loads
    plt.subplot(3, 2, 4)
    plt.plot(metrics['server_loads'])
    plt.title('Average Server Load')
    plt.xlabel('Episode')
    plt.ylabel('Load')
    plt.grid(True, alpha=0.3)
    
    # Plot epsilon decay
    plt.subplot(3, 2, 5)
    plt.plot(metrics['epsilon_values'])
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True, alpha=0.3)
    
    # Plot server selection distribution
    plt.subplot(3, 2, 6)
    if metrics['selected_servers']:
        server_dist = np.array(metrics['selected_servers'][-min(5, len(metrics['selected_servers'])):]).mean(axis=0)
        plt.bar(range(len(server_dist)), server_dist)
        plt.title('Recent Server Selection Distribution')
        plt.xlabel('Server ID')
        plt.ylabel('Selection Frequency')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'mec_gnn_metrics_episode_{episode}.png')
    plt.close()

def run_experiment():
    """Run full experiment with multiple configurations and comparative analysis"""
    # Configuration variations to test
    configs = [
        {"name": "Baseline", "hidden_dim": 64, "num_heads": 2, "num_layers": 3},
        {"name": "Enhanced", "hidden_dim": 128, "num_heads": 4, "num_layers": 4}
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nRunning experiment with configuration: {config['name']}")
        
        # Initialize environment
        env = MECEnvironment(num_edge_servers=10)
        
        # Initialize agent with configuration
        agent = MECGNNImprovedAgent(
            num_servers=env.num_edge_servers,
            node_features=4,
            edge_features=3,
            hidden_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            lr=0.0001
        )
        
        # Training parameters
        num_episodes = 500  # Reduced for quicker experiments
        max_steps = 100
        eval_frequency = 50
        
        metrics = {
            'rewards': [],
            'latencies': [],
            'eval_rewards': []
        }
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_latencies = []
            
            # Clear n-step buffer at start of episode
            agent.n_step_buffer.clear()
            
            for step in range(max_steps):
                # Select and perform action
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Store transition and train
                agent.store_transition(state, action, reward, next_state, done)
                agent.train()
                
                episode_latencies.append(info['total_latency'])
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Update metrics
            metrics['rewards'].append(episode_reward)
            metrics['latencies'].append(np.mean(episode_latencies))
            
            # Evaluation
            if episode % eval_frequency == 0 or episode == num_episodes - 1:
                eval_reward = evaluate_agent(agent, env, 10)
                metrics['eval_rewards'].append(eval_reward)
                
                print(f"Episode {episode}/{num_episodes}, "
                      f"Train Reward: {np.mean(metrics['rewards'][-20:]):.2f}, "
                      f"Eval Reward: {eval_reward:.2f}")
        
        # Store results
        results[config["name"]] = metrics
        
        # Save model
        agent.save_model(f'mec_gnn_model_{config["name"]}.pt')
    
    # Compare results
    compare_results(results)
    
    return results

def compare_results(results):
    """Compare and visualize results from different configurations"""
    plt.figure(figsize=(15, 10))
    
    # Plot evaluation rewards
    plt.subplot(2, 1, 1)
    for name, metrics in results.items():
        eval_episodes = range(0, 500, 50)
        plt.plot(eval_episodes, metrics['eval_rewards'], marker='o', label=name)
    
    plt.title('Evaluation Rewards Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Average Evaluation Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot latencies
    plt.subplot(2, 1, 2)
    for name, metrics in results.items():
        # Use moving average for smoother visualization
        window_size = 20
        smoothed_latencies = np.convolve(
            metrics['latencies'], 
            np.ones(window_size)/window_size, 
            mode='valid'
        )
        plt.plot(range(window_size-1, len(metrics['latencies'])), smoothed_latencies, label=name)
    
    plt.title('Average Latency Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Average Latency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mec_gnn_comparative_results.png')
    plt.show()

def visualize_server_efficiency(agent, env, num_scenarios=5):
    """Visualize how the agent selects servers based on efficiency metrics"""
    plt.figure(figsize=(15, 10))
    
    for i in range(num_scenarios):
        # Reset environment with new random state
        state = env.reset()
        
        # Extract state components
        server_speeds = state['server_speeds']
        server_loads = state['server_loads']
        network_conditions = state['network_conditions']
        server_distances = state['server_distances']
        
        # Calculate efficiency metrics
        processing_efficiency = server_speeds * (1 - server_loads)
        network_efficiency = network_conditions * (1 - server_distances)
        overall_efficiency = processing_efficiency * 0.7 + network_efficiency * 0.3
        
        # Get agent's Q-values
        with torch.no_grad():
            q_values, _ = agent.policy_net(state)
            q_values = q_values.cpu().numpy()
        
        # Select action
        action = agent.select_action(state, evaluate=True)
        
        # Plot results for this scenario
        plt.subplot(num_scenarios, 1, i+1)
        
        x = np.arange(env.num_edge_servers)
        width = 0.15
        
        plt.bar(x - 2*width, server_speeds, width, label='Server Speed')
        plt.bar(x - width, 1 - server_loads, width, label='Available Capacity')
        plt.bar(x, network_conditions, width, label='Network Quality')
        plt.bar(x + width, overall_efficiency, width, label='Overall Efficiency')
        plt.bar(x + 2*width, q_values, width, label='Q-Values')
        
        # Highlight selected server
        plt.axvline(x=action, color='r', linestyle='--', alpha=0.7)
        plt.text(action, 0.5, 'Selected', color='r', rotation=90, ha='right')
        
        plt.xlabel('Server ID')
        plt.ylabel('Metric Value')
        plt.title(f'Scenario {i+1}: Agent Decision Making')
        plt.xticks(x)
        
        if i == 0:
            plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('mec_gnn_server_selection.png')
    plt.show()

def analyze_edge_attention(agent, state):
    """Analyze and visualize the attention weights in the GNN model"""
    # Create graph from state
    data = agent.policy_net.create_graph_from_state(state)
    
    # Process node features through first layers 
    x = agent.policy_net.node_encoder(data.x)
    
    # Get attention weights from first GAT layer
    with torch.no_grad():
        # Here we would need to extract attention weights
        # This is a simplified visualization since we don't have direct 
        # access to attention weights without modifying the GATConv implementation
        
        # Instead, we'll visualize the graph structure
        edge_index = data.edge_index.cpu().numpy()
        
        # Create a networkx graph
        import networkx as nx
        G = nx.DiGraph()
        
        # Add nodes
        G.add_node(0, type='task', pos=(0, 0))
        for i in range(1, agent.num_servers + 1):
            G.add_node(i, type='server', pos=(i, 1))
        
        # Add edges
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            G.add_edge(src.item(), dst.item())
        
        # Draw graph
        plt.figure(figsize=(12, 8))
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw nodes
        task_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'task']
        server_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'server']
        
        nx.draw_networkx_nodes(G, pos, nodelist=task_nodes, node_color='r', node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=server_nodes, node_color='b', node_size=300, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        
        # Draw labels
        labels = {0: 'Task'}
        for i in range(1, agent.num_servers + 1):
            labels[i] = f'Server {i-1}'
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        plt.title('MEC Task Offloading Graph Structure')
        plt.axis('off')
        plt.savefig('mec_gnn_graph_structure.png')
        plt.show()

if __name__ == "__main__":
    # Run main training
    agent, metrics = train_improved_mec_gnn(render=True)
    
    # Optional: Run comparative experiment
    # results = run_experiment()
    
    # Visualize agent's decision making
    env = MECEnvironment(num_edge_servers=10)
    visualize_server_efficiency(agent, env)
    
    # Analyze graph structure
    state = env.reset()
    analyze_edge_attention(agent, state)