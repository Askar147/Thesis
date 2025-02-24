import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from collections import deque
import random
from DQN_second import MECEnvironment

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal information"""
    def __init__(self, d_model, max_seq_length=100):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MECTransformer(nn.Module):
    """Transformer for MEC task offloading decisions"""
    def __init__(self, state_dim, action_dim, seq_length=10, d_model=128, nhead=4, 
                 num_layers=3, dropout=0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_length = seq_length
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(state_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output heads
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim)
        )
        
    def forward(self, states):
        # Project and reshape states
        batch_size = states.size(0)
        x = states.view(batch_size, -1, self.state_dim)
        
        # Project to d_model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Get last sequence element for predictions
        x = x[:, -1]
        
        # Get policy and value predictions
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value

class TransformerReplayBuffer:
    """Replay buffer adapted for sequential data"""
    def __init__(self, capacity, seq_length):
        self.buffer = deque(maxlen=capacity)
        self.seq_length = seq_length
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # Ensure we have enough samples for a sequence
        valid_idx = len(self.buffer) - self.seq_length
        if valid_idx < batch_size:
            return None
        
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        # Sample starting indices
        start_indices = np.random.randint(0, valid_idx, size=batch_size)
        
        for start_idx in start_indices:
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            
            # Get sequence
            for i in range(self.seq_length):
                transition = self.buffer[start_idx + i]
                states.append(transition[0])
                actions.append(transition[1])
                rewards.append(transition[2])
                next_states.append(transition[3])
                dones.append(transition[4])
            
            batch_states.append(states)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_next_states.append(next_states)
            batch_dones.append(dones)
        
        # Convert to tensors
        return (
            torch.FloatTensor(batch_states),
            torch.LongTensor(batch_actions),
            torch.FloatTensor(batch_rewards),
            torch.FloatTensor(batch_next_states),
            torch.FloatTensor(batch_dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class MECTransformerAgent:
    """Transformer-based agent for MEC task offloading"""
    def __init__(self, state_dim, action_dim, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.seq_length = 10
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.min_replay_size = 1500
        
        # Add epsilon parameters for exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        
        # Initialize networks
        self.policy_net = MECTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            seq_length=self.seq_length
        ).to(device)
        
        self.target_net = MECTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            seq_length=self.seq_length
        ).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003)
        
        # Initialize replay buffer
        self.replay_buffer = TransformerReplayBuffer(100000, self.seq_length)
        
        # Initialize episode buffer for sequence building
        self.episode_buffer = []

    def forward(self, states):
        # Project and reshape states
        batch_size = states.size(0)
        x = states.view(batch_size, -1, self.state_dim)
        
        # Project to d_model dimension and apply dropout
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Get last sequence element for predictions
        x = x[:, -1]
        
        # Get policy and value predictions
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
        
    def select_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            # Add state to episode buffer
            self.episode_buffer.append(state)
            if len(self.episode_buffer) > self.seq_length:
                self.episode_buffer.pop(0)
            
            # Pad sequence if needed
            while len(self.episode_buffer) < self.seq_length:
                self.episode_buffer.insert(0, state)
            
            # Create sequence
            state_seq = torch.FloatTensor(self.episode_buffer).unsqueeze(0).to(self.device)
            
            # Get policy predictions
            policy, _ = self.policy_net(state_seq)
            return policy.max(1)[1].item()
    
    def train(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        # Sample from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)
        
        # Get current Q values
        policy_output, current_value = self.policy_net(state_batch)
        current_q = policy_output.gather(1, action_batch[:, -1].unsqueeze(1))
        
        # Get target Q values
        with torch.no_grad():
            next_policy_output, next_value = self.target_net(next_state_batch)
            target_q = reward_batch[:, -1].unsqueeze(1) + \
                      (1 - done_batch[:, -1].unsqueeze(1)) * self.gamma * next_value
        
        # Compute loss
        value_loss = nn.MSELoss()(current_value, target_q)
        policy_loss = nn.CrossEntropyLoss()(policy_output, action_batch[:, -1])
        loss = value_loss + policy_loss
        
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
        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

def train_mec_transformer():
    """Training loop for Transformer in MEC environment"""
    # Initialize environment
    env = MECEnvironment(num_edge_servers=10)
    
    # Calculate state size
    state_size = (1 +  # task_size
                env.num_edge_servers +  # server_speeds
                env.num_edge_servers +  # server_loads
                env.num_edge_servers +  # network_conditions
                env.num_edge_servers)   # server_distances
    action_size = env.num_edge_servers
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Initialize agent
    agent = MECTransformerAgent(state_size, action_size)
    
    # Training parameters
    num_episodes = 2000
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
        obs = env.reset()
        state = np.concatenate([
            obs['task_size'],
            obs['server_speeds'],
            obs['server_loads'],
            obs['network_conditions'],
            obs['server_distances']
        ])
        
        episode_reward = 0
        episode_losses = []
        episode_latencies = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_obs, reward, done, info = env.step(action)
            
            # Process next state
            next_state = np.concatenate([
                next_obs['task_size'],
                next_obs['server_speeds'],
                next_obs['server_loads'],
                next_obs['network_conditions'],
                next_obs['server_distances']
            ])
            
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
        metrics['server_loads'].append(np.mean(next_obs['server_loads']))
        
        if episode_losses:
            metrics['losses'].append(np.mean(episode_losses))
        
        # Calculate running average
        if episode >= eval_frequency:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            metrics['avg_rewards'].append(avg_reward)
        
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)


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
    agent, metrics = train_mec_transformer()