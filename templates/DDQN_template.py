import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Note: VECEnvironment remains the same as in DQN implementation

class DDQNNetwork(nn.Module):
    """Double Deep Q-Network"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DDQNNetwork, self).__init__()
        
        # Shared feature layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        features = self.feature_layers(x)
        
        advantage = self.advantage_stream(features)
        value = self.value_stream(features)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class DDQNAgent:
    """DDQN Agent for VEC task offloading"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # DDQN hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001  # Smaller learning rate than DQN
        self.batch_size = 64
        self.tau = 0.001  # Soft update parameter
        
        # Create Q-Networks
        self.online_network = DDQNNetwork(state_size, action_size)
        self.target_network = DDQNNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.online_network.state_dict())
        
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        
        # Replay buffer with prioritization
        self.replay_buffer = PrioritizedReplayBuffer(100000)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online_network.to(self.device)
        self.target_network.to(self.device)
        
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_network(state)
            return q_values.argmax().item()
    
    def train(self):
        """Train the agent using double Q-learning"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch with priorities
        batch, weights, indices = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Get current Q values
        current_q_values = self.online_network(states).gather(1, actions.unsqueeze(1))
        
        # DDQN: Use online network to select action and target network to evaluate it
        with torch.no_grad():
            # Select actions using online network
            next_actions = self.online_network(next_states).argmax(1).unsqueeze(1)
            # Evaluate actions using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute TD errors for prioritized replay
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
        
        # Update priorities
        for idx, error in zip(indices, td_errors):
            self.replay_buffer.update_priority(idx, error[0])
        
        # Compute loss with importance sampling weights
        loss = (weights * (current_q_values - target_q_values) ** 2).mean()
        
        # Optimize the online network
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update_target_network()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def _soft_update_target_network(self):
        """Soft update target network parameters"""
        for target_param, online_param in zip(
            self.target_network.parameters(), 
            self.online_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = 0.001  # Beta increment per sampling
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer with max priority"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def sample(self, batch_size):
        """Sample batch with priorities"""
        buffer_len = len(self.buffer)
        
        # Calculate probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(buffer_len, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (buffer_len * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get samples
        batch = [self.buffer[idx] for idx in indices]
        
        return batch, weights, indices
    
    def update_priority(self, idx, priority):
        """Update priority of experience"""
        self.priorities[idx] = priority + 1e-5  # Small constant to prevent zero priority
    
    def __len__(self):
        return len(self.buffer)

def train_ddqn():
    """Main training loop"""
    env = VECEnvironment()
    
    # Calculate state size from observation space
    state_size = (env.observation_space['vehicle_states'].shape[0] + 
                 env.observation_space['server_states'].shape[0])
    action_size = env.action_space.n
    
    agent = DDQNAgent(state_size, action_size)
    
    num_episodes = 1000
    max_steps_per_episode = 1000
    
    for episode in range(num_episodes):
        state = env.reset()
        state = np.concatenate([state['vehicle_states'], state['server_states']])
        episode_reward = 0
        episode_loss = 0
        
        for step in range(max_steps_per_episode):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Flatten the next state
            next_state = np.concatenate([
                next_state['vehicle_states'], 
                next_state['server_states']
            ])
            
            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train the agent
            loss = agent.train()
            if loss is not None:
                episode_loss += loss
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Logging
        print(f"Episode {episode + 1}")
        print(f"Total Reward: {episode_reward}")
        print(f"Average Loss: {episode_loss / (step + 1)}")
        print(f"Final Epsilon: {agent.epsilon}")
        print("------------------------")

if __name__ == "__main__":
    train_ddqn()