import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data, Batch
import numpy as np
from collections import deque
import random



import gym
from gym import spaces

class MECEnvironment(gym.Env):
    """Enhanced MEC Environment with realistic latency components and improved server load dynamics"""
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
        
        # Observation space includes task size, server speeds, loads, network conditions, and distances.
        self.observation_space = spaces.Dict({
            'task_size': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'server_speeds': spaces.Box(low=np.zeros(num_edge_servers), high=np.ones(num_edge_servers), dtype=np.float32),
            'server_loads': spaces.Box(low=np.zeros(num_edge_servers), high=np.ones(num_edge_servers), dtype=np.float32),
            'network_conditions': spaces.Box(low=np.zeros(num_edge_servers), high=np.ones(num_edge_servers), dtype=np.float32),
            'server_distances': spaces.Box(low=np.zeros(num_edge_servers), high=np.ones(num_edge_servers), dtype=np.float32)
        })
        
        # Initialize server characteristics with improved ranges
        self.server_speeds = np.random.uniform(0.7, 1.0, num_edge_servers)   # Higher minimum speed
        self.server_distances = np.random.uniform(0.1, 0.8, num_edge_servers)  # Lower maximum distance
        self.bandwidth_up = np.random.uniform(0.6, 1.0, num_edge_servers)      # Better uplink bandwidth
        self.bandwidth_down = np.random.uniform(0.7, 1.0, num_edge_servers)    # Better downlink bandwidth
        
        # Scaling factors for various latency components (adjustable)
        self.uplink_scale = 0.8
        self.prop_scale = 0.05
        self.downlink_scale = 0.8
        self.queue_factor = 1.2
        self.decay_factor = 0.95  # For exponential decay of non-selected server loads
        
        self.reset()
    
    def reset(self):
        """Reset environment state with new task and updated loads/conditions"""
        self.current_task_size = np.random.uniform(0.2, 0.8)  # Smaller maximum task size
        self.server_loads = np.random.uniform(0.1, 0.3, self.num_edge_servers)  # Lower initial loads
        self.network_conditions = np.random.uniform(0.8, 1.0, self.num_edge_servers)  # Better network conditions
        return self._get_observation()
    
    def _calculate_total_latency(self, server_idx):
        """Calculate total latency including uplink, propagation, processing, downlink, and queuing delays"""
        # 1. Uplink transmission delay
        uplink_delay = (self.current_task_size / self.bandwidth_up[server_idx]) * \
                       (1 / self.network_conditions[server_idx]) * self.uplink_scale
        
        # 2. Propagation delay based on distance
        prop_delay = self.server_distances[server_idx] * self.prop_scale
        
        # 3. Processing delay (affected by server load)
        effective_speed = self.server_speeds[server_idx] * (1 - self.server_loads[server_idx])
        processing_delay = self.current_task_size / max(effective_speed, 0.1)
        
        # 4. Downlink transmission delay (assume result is a fraction of input size)
        result_size = self.current_task_size * 0.05
        downlink_delay = (result_size / self.bandwidth_down[server_idx]) * \
                         (1 / self.network_conditions[server_idx]) * self.downlink_scale
        
        # 5. Queuing delay (scaled by server load)
        queue_delay = self.server_loads[server_idx] * processing_delay * self.queue_factor
        
        total_delay = uplink_delay + prop_delay + processing_delay + downlink_delay + queue_delay
        return total_delay
    
    def step(self, action):
        """Take an action, update state, and return observation, reward, done flag, and info."""
        if self.continuous_action:
            action_probs = F.softmax(torch.FloatTensor(action), dim=0).numpy()
            selected_server = np.argmax(action_probs)
        else:
            selected_server = action
        
        # Calculate latency for the chosen server
        total_latency = self._calculate_total_latency(selected_server)
        
        # Compute a normalized latency for reward scaling
        normalized_latency = total_latency / 5.0  # Adjust scale as needed
        base_reward = -np.tanh(normalized_latency)  # Reward between -1 and 1
        
        # Add bonus for selecting a server with the highest effective speed
        available_speeds = self.server_speeds * (1 - self.server_loads)
        if selected_server == np.argmax(available_speeds):
            base_reward += 0.3
        elif available_speeds[selected_server] >= np.percentile(available_speeds, 75):
            base_reward += 0.1
        
        # Penalize if server load is very high
        if self.server_loads[selected_server] > 0.8:
            base_reward -= 0.2
        
        # Update environment state
        self._update_server_loads(selected_server)
        self._update_network_conditions()
        self.current_task_size = np.random.uniform(0.2, 0.8)
        
        observation = self._get_observation()
        info = {
            'selected_server': selected_server,
            'server_load': self.server_loads[selected_server],
            'network_quality': self.network_conditions[selected_server],
            'total_latency': total_latency,
            'effective_speed': available_speeds[selected_server]
        }
        
        return observation, base_reward, False, info
    
    def _update_server_loads(self, selected_server):
        """Update server loads after task assignment using exponential decay for non-selected servers"""
        # Increase load for the selected server
        self.server_loads[selected_server] = min(
            self.server_loads[selected_server] + self.current_task_size * 0.1,
            1.0
        )
        # Apply exponential decay to other servers
        for i in range(self.num_edge_servers):
            if i != selected_server:
                self.server_loads[i] = max(self.server_loads[i] * self.decay_factor, 0.1)
    
    def _update_network_conditions(self):
        """Update network conditions with random fluctuations"""
        fluctuation = np.random.uniform(-0.1, 0.1, self.num_edge_servers)
        self.network_conditions += fluctuation
        self.network_conditions = np.clip(self.network_conditions, 0.3, 1.0)
    
    def _get_observation(self):
        """Construct the observation dictionary"""
        return {
            'task_size': np.array([self.current_task_size], dtype=np.float32),
            'server_speeds': self.server_speeds.astype(np.float32),
            'server_loads': self.server_loads.astype(np.float32),
            'network_conditions': self.network_conditions.astype(np.float32),
            'server_distances': self.server_distances.astype(np.float32)
        }
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data, Batch
import numpy as np
from collections import deque
import random

class EdgeConv(MessagePassing):
    """Custom edge convolution layer for MEC"""
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max')
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr=None):
        tmp = torch.cat([x_i, x_j], dim=1)
        return self.mlp(tmp)

class MECGraphNet(nn.Module):
    """Graph Neural Network for MEC task offloading"""
    def __init__(self, node_features, edge_features, num_servers, 
                 hidden_dim=64, num_layers=3, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(MECGraphNet, self).__init__()
        
        # Device setup
        self.device = device
        
        # Dimensions
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_servers = num_servers
        self.hidden_dim = hidden_dim
        
        # Node feature processing
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        ).to(device)
        
        # Edge feature processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        ).to(device)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            EdgeConv(hidden_dim, hidden_dim).to(device) for _ in range(num_layers)
        ])
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim).to(device) for _ in range(num_layers)
        ])
        
        # Q-value predictor
        self.q_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_servers)
        ).to(device)
        
        # Value predictor
        self.value_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        ).to(device)
    
    def create_graph_from_state(self, state_dict):
        """Convert MEC environment state to graph"""
        # Extract state components and convert to tensors on the correct device
        task_size = torch.FloatTensor(state_dict['task_size']).to(self.device)
        server_speeds = torch.FloatTensor(state_dict['server_speeds']).to(self.device)
        server_loads = torch.FloatTensor(state_dict['server_loads']).to(self.device)
        network_conditions = torch.FloatTensor(state_dict['network_conditions']).to(self.device)
        server_distances = torch.FloatTensor(state_dict['server_distances']).to(self.device)
        
        # Create task node features
        # Pad task features to match server feature dimension (4 features per server)
        padding = torch.zeros(3, device=self.device)  # We have 1 task feature, need 3 more to match
        task_features = torch.cat([task_size, padding]).unsqueeze(0)  # Shape: [1, 4]
        
        # Create server node features
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
        
        # Add edges from task to each server
        for i in range(self.num_servers):
            # Task -> Server
            edge_index.append([0, i + 1])
            edge_attr.append([
                network_conditions[i].item(),
                server_distances[i].item(),
                server_loads[i].item()
            ])
            # Server -> Task
            edge_index.append([i + 1, 0])
            edge_attr.append([
                network_conditions[i].item(),
                server_distances[i].item(),
                server_loads[i].item()
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
        
        # Create edges (fully connected between task and servers)
        edge_index = []
        edge_attr = []
        
        # Add edges from task to each server
        for i in range(self.num_servers):
            # Task -> Server
            edge_index.append([0, i + 1])
            edge_attr.append([
                network_conditions[i].item(),
                server_distances[i].item(),
                server_loads[i].item()
            ])
            # Server -> Task
            edge_index.append([i + 1, 0])
            edge_attr.append([
                network_conditions[i].item(),
                server_distances[i].item(),
                server_loads[i].item()
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
        
        # Create task node features
        task_features = torch.cat([
            task_size,
            torch.zeros(self.num_servers - 1)  # Padding to match server features
        ]).unsqueeze(0)
        
        # Create server node features
        server_features = torch.stack([
            torch.cat([
                server_speeds[i].unsqueeze(0),
                server_loads[i].unsqueeze(0),
                network_conditions[i].unsqueeze(0),
                server_distances[i].unsqueeze(0)
            ]) for i in range(self.num_servers)
        ])
        
        # Combine all node features
        node_features = torch.cat([task_features, server_features])
        
        # Create edges (fully connected between task and servers)
        edge_index = []
        edge_attr = []
        
        # Add edges from task to each server
        for i in range(self.num_servers):
            # Task -> Server
            edge_index.append([0, i + 1])
            edge_attr.append([
                network_conditions[i],
                server_distances[i],
                server_loads[i]
            ])
            # Server -> Task
            edge_index.append([i + 1, 0])
            edge_attr.append([
                network_conditions[i],
                server_distances[i],
                server_loads[i]
            ])
        
        # Convert to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
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
        
        # Process edge features
        edge_attr = self.edge_encoder(data.edge_attr)
        
        # Apply graph convolutions with residual connections
        for conv, attention in zip(self.conv_layers, self.attention_layers):
            x_conv = conv(x, data.edge_index, edge_attr)
            x_attention = attention(x, data.edge_index)
            x = x + x_conv + x_attention
            x = F.relu(x)
        
        # Get task node representation (first node)
        task_repr = x[0]
        
        # Get Q-values and value
        q_values = self.q_predictor(task_repr)
        value = self.value_predictor(task_repr)
        
        return q_values, value

class GNNReplayBuffer:
    """Replay buffer for GNN-based agent"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class MECGNNAgent:
    """GNN-based agent for MEC task offloading"""
    def __init__(self, num_servers, node_features=4, edge_features=3,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.num_servers = num_servers
        
        # Initialize networks
        self.policy_net = MECGraphNet(
            node_features=node_features,
            edge_features=edge_features,
            num_servers=num_servers,
            device=device
        )
        
        self.target_net = MECGraphNet(
            node_features=node_features,
            edge_features=edge_features,
            num_servers=num_servers,
            device=device
        )
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.0003)
        
        # Initialize replay buffer
        self.replay_buffer = GNNReplayBuffer(100000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.min_replay_size = 1000
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
    
    def select_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.num_servers)
        
        with torch.no_grad():
            q_values, _ = self.policy_net(state)
            return q_values.max(0)[1].item()
    
    def train(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        # Sample from replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # Process batch
        state_batch = batch[0]  # List of state dictionaries
        action_batch = torch.tensor(batch[1], device=self.device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=self.device)
        next_state_batch = batch[3]  # List of state dictionaries
        done_batch = torch.tensor(batch[4], dtype=torch.float32, device=self.device)
        
        # Get current Q values
        current_q_values = []
        for state in state_batch:
            q_values, _ = self.policy_net(state)
            current_q_values.append(q_values)
        current_q_values = torch.stack(current_q_values)
        current_q = current_q_values.gather(1, action_batch.unsqueeze(1))
        
        # Get target Q values
        target_q = []
        with torch.no_grad():
            for next_state in next_state_batch:
                q_values, value = self.target_net(next_state)
                target_q.append(value)
            target_q = torch.stack(target_q).to(self.device)
            target_q = reward_batch.unsqueeze(1) + \
                      (1.0 - done_batch.unsqueeze(1)) * self.gamma * target_q
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        for target_param, policy_param in zip(self.target_net.parameters(),
                                            self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        for target_param, policy_param in zip(self.target_net.parameters(),
                                            self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

def train_mec_gnn():
    """Training loop for GNN in MEC environment"""
    # Initialize environment
    env = MECEnvironment(num_edge_servers=10)
    
    # Initialize agent
    agent = MECGNNAgent(num_servers=env.num_edge_servers)
    
    # Training parameters
    num_episodes = 1000
    max_steps = 100
    eval_frequency = 50
    
    # Metrics tracking
    metrics = {
        'rewards': [],
        'losses': [],
        'latencies': [],
        'server_loads': [],
        'avg_rewards': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        episode_latencies = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition and train
            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.train()
            
            if loss is not None:
                episode_losses.append(loss)
            
            episode_latencies.append(info['total_latency'])
            state = next_state
            episode_reward += reward
        
        # Update metrics
        metrics['rewards'].append(episode_reward)
        metrics['latencies'].append(np.mean(episode_latencies))
        metrics['server_loads'].append(np.mean(state['server_loads']))
        
        if episode_losses:
            metrics['losses'].append(np.mean(episode_losses))
        
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
    agent, metrics = train_mec_gnn()