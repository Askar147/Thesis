import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import gym
from gym import spaces
from collections import deque
import random
import os
import time
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau  # Add this line



class MECEnvironment(gym.Env):
    """Enhanced MEC Environment with more stable dynamics and reward structure"""
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
        
        # Initialize server characteristics with improved ranges
        self.server_speeds = np.random.uniform(0.7, 1.0, num_edge_servers)   # Higher minimum speed
        self.server_distances = np.random.uniform(0.1, 0.8, num_edge_servers)  # Lower maximum distance
        self.bandwidth_up = np.random.uniform(0.6, 1.0, num_edge_servers)      # Better uplink bandwidth
        self.bandwidth_down = np.random.uniform(0.7, 1.0, num_edge_servers)    # Better downlink bandwidth
        
        # Scaling factors for various latency components
        self.uplink_scale = 0.6    # Reduced for less extreme penalties
        self.prop_scale = 0.03     # Reduced for less extreme penalties
        self.downlink_scale = 0.6  # Reduced for less extreme penalties
        self.queue_factor = 0.8    # Reduced queue impact
        
        # Keep track of latency history for normalization
        self.latency_history = deque(maxlen=100)
        self.prev_fluctuation = np.zeros(num_edge_servers)
        
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
        effective_speed = self.server_speeds[server_idx] * (1 - 0.8 * self.server_loads[server_idx])
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
        self.latency_history.append(total_latency)
        
        # Compute reward components
        # 1. Latency component with improved normalization
        if len(self.latency_history) > 10:
            # Normalize latency against recent history to provide more stable learning signal
            avg_latency = np.mean(self.latency_history)
            std_latency = np.std(self.latency_history) + 0.1  # Add small constant to avoid division by zero
            
            # Z-score relative to recent history (how much better/worse than average)
            latency_score = (total_latency - avg_latency) / std_latency
            
            # Cap extreme values and convert to reward (higher is better)
            latency_reward = -1.0 - np.clip(latency_score, -1.0, 1.0)  # Range: -2.0 to 0.0
        else:
            # Early in training, use simpler scaling
            latency_reward = -total_latency / 2.0
        
        # 2. Load balancing incentive (small bonus for using less loaded servers)
        load_balance_reward = -0.2 * self.server_loads[selected_server]  # Range: -0.2 to 0.0
        
        # 3. Server selection incentive
        effective_speeds = self.server_speeds * (1 - 0.8 * self.server_loads)
        optimal_server = np.argmax(effective_speeds)
        
        # Bonus for selecting optimal or near-optimal server
        if selected_server == optimal_server:
            selection_reward = 0.2  # Bonus for optimal selection
        else:
            # Partial credit based on how close to optimal the selection was
            selection_rank = np.sum(effective_speeds > effective_speeds[selected_server])
            selection_reward = 0.2 * (1.0 - selection_rank / self.num_edge_servers)
        
        # Final reward combines components (dominated by latency)
        reward = latency_reward + load_balance_reward + selection_reward
        
        # Update environment state
        self._update_server_loads(selected_server)
        self._update_network_conditions()
        self.current_task_size = np.random.uniform(0.2, 0.8)
        
        observation = self._get_observation()
        info = {
            'selected_server': selected_server,
            'server_load': self.server_loads[selected_server],
            'total_latency': total_latency,
            'latency_reward': latency_reward,
            'load_balance_reward': load_balance_reward,
            'selection_reward': selection_reward
        }
        
        return observation, reward, False, info
    
    def _update_server_loads(self, selected_server):
        """Update server loads with more stable dynamics"""
        # Increase load for the selected server (more gradual increase)
        self.server_loads[selected_server] = min(
            self.server_loads[selected_server] + self.current_task_size * 0.08,
            0.95  # Cap at 95% to prevent complete saturation
        )
        
        # Apply smoother decay to other servers
        for i in range(self.num_edge_servers):
            if i != selected_server:
                # Linear decay instead of exponential for more stability
                self.server_loads[i] = max(self.server_loads[i] - 0.01, 0.1)
    
    def _update_network_conditions(self):
        """Update network conditions with smoother, temporally correlated fluctuations"""
        # Use much smaller fluctuations
        fluctuation = np.random.uniform(-0.03, 0.03, self.num_edge_servers)
        
        # Make fluctuations temporally correlated (smoother changes)
        self.prev_fluctuation = 0.7 * self.prev_fluctuation + 0.3 * fluctuation
        
        # Apply the smoothed fluctuations
        self.network_conditions += self.prev_fluctuation
        
        # Ensure values stay in reasonable range
        self.network_conditions = np.clip(self.network_conditions, 0.6, 1.0)  # Higher minimum for more stability
    
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
    """Improved Transformer for MEC task offloading decisions"""
    def __init__(self, state_dim, action_dim, seq_length=16, d_model=128, nhead=4, 
                 num_layers=3, dropout=0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_length = seq_length
        self.d_model = d_model
        
        # Input normalization
        self.state_norm = nn.LayerNorm(state_dim)
        
        # Input projection with layer normalization
        self.input_projection = nn.Linear(state_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
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
        
        # Output heads with improved architecture
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights using appropriate initialization methods"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.kaiming_normal_(param)  # Better init for ReLU
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(self, states):
        # Project and reshape states
        batch_size = states.size(0)
        x = states.view(batch_size, -1, self.state_dim)
        
        # Normalize input states
        x = self.state_norm(x)
        
        # Project to d_model dimension and apply normalization
        x = self.input_projection(x)
        x = self.input_norm(x)
        
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


class SimpleReplayBuffer:
    """Simplified replay buffer for better stability"""
    def __init__(self, capacity, seq_length):
        self.buffer = deque(maxlen=capacity)
        self.seq_length = seq_length
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # Ensure we have enough samples for a sequence
        if len(self.buffer) < self.seq_length + batch_size:
            return None
        
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        # Sample starting indices
        valid_idx = len(self.buffer) - self.seq_length
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
            torch.FloatTensor(np.array(batch_states)),
            torch.LongTensor(np.array(batch_actions)),
            torch.FloatTensor(np.array(batch_rewards)),
            torch.FloatTensor(np.array(batch_next_states)),
            torch.FloatTensor(np.array(batch_dones))
        )
    
    def __len__(self):
        return len(self.buffer)


class MECTransformerAgent:
    """Improved Transformer-based agent for MEC task offloading"""
    def __init__(self, state_dim, action_dim, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Improved hyperparameters
        self.seq_length = 16  # Longer sequence length to capture temporal patterns
        self.gamma = 0.99     # Standard discount factor
        self.tau = 0.005      # Slower target network updates for stability
        self.batch_size = 64  # Smaller batch size for more frequent updates
        self.min_replay_size = 3000  # Larger minimum buffer for better initialization
        
        # Exploration parameters with slower decay
        self.epsilon = 1.0
        self.epsilon_min = 0.15
        self.epsilon_decay = 0.9992  # Much slower decay
        
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
        
        # Initialize optimizer with better learning rate management
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003, weight_decay=1e-5)
        self.scheduler = StepLR(self.optimizer, step_size=400, gamma=0.5)  # Halve LR every 400 episodes
        
        # Initialize replay buffer
        self.replay_buffer = SimpleReplayBuffer(100000, self.seq_length)
        
        # Initialize episode buffer for sequence building
        self.episode_buffer = []
        
        # Performance tracking
        self.training_rewards = []
        self.update_count = 0
        
    def select_action(self, state, evaluate=False):
        """Select action with epsilon-greedy policy"""
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
            
            # Convert to tensor
            state_seq = np.array(self.episode_buffer)
            state_seq = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
            
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
        
        # Get target Q values with double Q-learning
        with torch.no_grad():
            # Get actions from policy network
            next_policy_output, _ = self.policy_net(next_state_batch)
            next_actions = next_policy_output.max(1)[1].unsqueeze(1)
            
            # Get Q-values from target network for those actions
            next_target_policy, next_target_value = self.target_net(next_state_batch)
            next_q = next_target_policy.gather(1, next_actions)
            
            # Compute target
            target_q = reward_batch[:, -1].unsqueeze(1) + \
                      (1 - done_batch[:, -1].unsqueeze(1)) * self.gamma * next_q
        
        # Compute loss with Huber loss for stability
        value_loss = F.smooth_l1_loss(current_value, target_q)
        policy_loss = F.smooth_l1_loss(current_q, target_q.detach())
        loss = value_loss + policy_loss
        
        # Gradient clipping to prevent exploding gradients
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network periodically
        if self.update_count % 10 == 0:  # Less frequent target updates
            for target_param, policy_param in zip(self.target_net.parameters(), 
                                                 self.policy_net.parameters()):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1 - self.tau) * target_param.data
                )
        
        self.update_count += 1
        return loss.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, path):
        """Save the policy network"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """Load the policy network"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


def train_mec_transformer():
    """Training loop with better monitoring and stability"""
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
    
    # Initialize agent with improved parameters
    agent = MECTransformerAgent(state_size, action_size)
    
    # Override key hyperparameters
    agent.epsilon = 1.0
    agent.epsilon_min = 0.15  # Higher minimum exploration
    agent.epsilon_decay = 0.9992  # Much slower decay (~1000 episodes to reach minimum)
    
    # Training parameters
    num_episodes = 2000
    max_steps = 100
    eval_frequency = 20  # More frequent evaluation
    
    # Early stopping parameters with more patience
    early_stop_patience = 50  # Number of evaluations without improvement
    best_reward = -float('inf')
    no_improvement_count = 0
    
    # Larger evaluation set for more reliable metrics
    eval_episodes = 20  
    
    # Create evaluation function
    def evaluate_agent():
        eval_rewards = []
        eval_latencies = []
        
        for _ in range(eval_episodes):
            obs = env.reset()
            state = np.concatenate([
                obs['task_size'],
                obs['server_speeds'],
                obs['server_loads'],
                obs['network_conditions'],
                obs['server_distances']
            ])
            
            episode_reward = 0
            episode_latencies = []
            agent.episode_buffer = []  # Clear episode buffer
            
            for step in range(max_steps):
                action = agent.select_action(state, evaluate=True)
                next_obs, reward, done, info = env.step(action)
                
                next_state = np.concatenate([
                    next_obs['task_size'],
                    next_obs['server_speeds'],
                    next_obs['server_loads'],
                    next_obs['network_conditions'],
                    next_obs['server_distances']
                ])
                
                episode_latencies.append(info['total_latency'])
                state = next_state
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            eval_latencies.append(np.mean(episode_latencies))
        
        return np.mean(eval_rewards), np.mean(eval_latencies)
    
    # Track metrics
    metrics = {
        'episode_rewards': [],
        'eval_rewards': [],
        'eval_latencies': [],
        'train_latencies': [],
        'epsilon_values': [],
        'learning_rates': []
    }
    
    # Create directory for saving models
    os.makedirs('models', exist_ok=True)
    
    # Start training timer
    start_time = time.time()
    
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
        episode_latencies = []
        agent.episode_buffer = []  # Clear episode buffer at start
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_obs, reward, done, info = env.step(action)
            
            next_state = np.concatenate([
                next_obs['task_size'],
                next_obs['server_speeds'],
                next_obs['server_loads'],
                next_obs['network_conditions'],
                next_obs['server_distances']
            ])
            
            # Store transition and train
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train()
            
            episode_latencies.append(info['total_latency'])
            state = next_state
            episode_reward += reward
        
        # Update epsilon with slower decay
        agent.update_epsilon()
        
        # Step the learning rate scheduler (only every 5 episodes)
        if episode % 5 == 0:
            agent.scheduler.step()
        
        # Track metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['train_latencies'].append(np.mean(episode_latencies))
        metrics['epsilon_values'].append(agent.epsilon)
        metrics['learning_rates'].append(agent.optimizer.param_groups[0]['lr'])
        
        # Evaluate and print progress
        if episode % eval_frequency == 0:
            eval_reward, eval_latency = evaluate_agent()
            metrics['eval_rewards'].append(eval_reward)
            metrics['eval_latencies'].append(eval_latency)
            
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print(f"Episode {episode}/{num_episodes} [{int(hours)}h {int(minutes)}m {int(seconds)}s] | "
                  f"Train Reward: {episode_reward:.2f} | "
                  f"Train Latency: {np.mean(episode_latencies):.2f} | "
                  f"Eval Reward: {eval_reward:.2f} | "
                  f"Eval Latency: {eval_latency:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"LR: {agent.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check for early stopping (based on evaluation reward)
            if eval_reward > best_reward:
                best_reward = eval_reward
                no_improvement_count = 0
                # Save best model
                agent.save_model("models/best_mec_transformer.pt")
                print(f"New best model saved with reward: {best_reward:.2f}")
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= early_stop_patience:
                print(f"Early stopping at episode {episode}")
                break
        
        # Save checkpoint every 100 episodes
        if episode % 100 == 0 and episode > 0:
            agent.save_model(f"models/mec_transformer_ep{episode}.pt")
    
    # Load best model for final evaluation
    agent.load_model("models/best_mec_transformer.pt")
    final_reward, final_latency = evaluate_agent()
    print(f"Final evaluation - Reward: {final_reward:.2f}, Latency: {final_latency:.2f}")
    
    return agent, metrics


if __name__ == "__main__":
    agent, metrics = train_mec_transformer()