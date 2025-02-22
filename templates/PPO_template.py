import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import gym

class ActorCritic(nn.Module):
    """Combined actor-critic network"""
    def __init__(self, state_dim, action_dim, continuous_action=True):
        super(ActorCritic, self).__init__()
        
        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        if continuous_action:
            self.actor_mean = nn.Linear(256, action_dim)
            self.actor_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(256, action_dim)
        
        # Critic head (value network)
        self.critic = nn.Linear(256, 1)
        
        self.continuous_action = continuous_action
    
    def forward(self, state):
        features = self.shared(state)
        
        # Get value
        value = self.critic(features)
        
        # Get action distribution
        if self.continuous_action:
            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_std)
            dist = Normal(action_mean, action_std)
        else:
            action_logits = self.actor(features)
            dist = Categorical(logits=action_logits)
        
        return dist, value

class PPOMemory:
    """Memory buffer for PPO"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def push(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self):
        return (
            torch.FloatTensor(self.states),
            torch.FloatTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.dones)
        )
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

class PPOAgent:
    """PPO Agent for VEC task offloading"""
    def __init__(self, state_dim, action_dim, continuous_action=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.value_clip_epsilon = 0.2
        self.c1 = 1.0  # Value loss coefficient
        self.c2 = 0.01  # Entropy coefficient
        
        # Training parameters
        self.batch_size = 64
        self.n_epochs = 10
        self.learning_rate = 3e-4
        
        # Initialize actor-critic network
        self.ac_network = ActorCritic(state_dim, action_dim, continuous_action).to(self.device)
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=self.learning_rate)
        
        # Initialize memory
        self.memory = PPOMemory()
        
        self.continuous_action = continuous_action
    
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            dist, value = self.ac_network(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            if self.continuous_action:
                log_prob = log_prob.sum(dim=-1)
        
        return (
            action.cpu().numpy()[0],
            value.cpu().numpy()[0],
            log_prob.cpu().numpy()
        )
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        
        return torch.FloatTensor(advantages)
    
    def train(self, next_value):
        """Update policy and value function"""
        # Get data from memory
        states, actions, rewards, values, old_log_probs, dones = self.memory.get()
        
        # Compute advantages and returns
        advantages = self.compute_gae(rewards, values, dones, next_value)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors and move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        values = values.to(self.device)
        
        # PPO training loop
        for _ in range(self.n_epochs):
            # Generate random mini-batches
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_states = states[start_idx:end_idx]
                batch_actions = actions[start_idx:end_idx]
                batch_old_log_probs = old_log_probs[start_idx:end_idx]
                batch_advantages = advantages[start_idx:end_idx]
                batch_returns = returns[start_idx:end_idx]
                batch_values = values[start_idx:end_idx]
                
                # Get current policy distribution and value
                dist, value = self.ac_network(batch_states)
                
                # Get current log probabilities
                curr_log_probs = dist.log_prob(batch_actions)
                if self.continuous_action:
                    curr_log_probs = curr_log_probs.sum(dim=-1)
                
                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                # Calculate value loss with clipping
                value_pred = value.squeeze()
                value_clipped = batch_values + torch.clamp(
                    value_pred - batch_values,
                    -self.value_clip_epsilon,
                    self.value_clip_epsilon
                )
                value_loss = torch.max(
                    (value_pred - batch_returns) ** 2,
                    (value_clipped - batch_returns) ** 2
                ).mean()
                
                # Calculate entropy bonus
                entropy = dist.entropy().mean()
                
                # Calculate total loss
                policy_loss = -torch.min(surr1, surr2).mean()
                total_loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
                
                # Update network
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), 0.5)
                self.optimizer.step()
        
        # Clear memory after update
        self.memory.clear()

def train_ppo():
    """Main training loop"""
    env = VECEnvironment()  # Your custom environment
    
    # Calculate state and action dimensions
    state_dim = (env.observation_space['vehicle_states'].shape[0] + 
                env.observation_space['server_states'].shape[0])
    action_dim = env.action_space.n
    continuous_action = isinstance(env.action_space, gym.spaces.Box)
    
    # Initialize agent
    agent = PPOAgent(state_dim, action_dim, continuous_action)
    
    # Training parameters
    num_episodes = 1000
    max_steps_per_episode = 1000
    update_timestep = 2048  # Update policy every n timesteps
    
    timestep = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        state = np.concatenate([state['vehicle_states'], state['server_states']])
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            # Select action
            action, value, log_prob = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            next_state = np.concatenate([
                next_state['vehicle_states'],
                next_state['server_states']
            ])
            
            # Store experience
            agent.memory.push(state, action, reward, value, log_prob, done)
            
            timestep += 1
            episode_reward += reward
            state = next_state
            
            # Update if its time
            if timestep % update_timestep == 0:
                # Get value of next state for GAE
                with torch.no_grad():
                    next_state = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                    _, next_value = agent.ac_network(next_state)
                    next_value = next_value.cpu().numpy()[0]
                
                # Update policy
                agent.train(next_value)
                timestep = 0
            
            if done:
                break
        
        print(f"Episode {episode + 1}, Total Reward: {episode_reward}")

if __name__ == "__main__":
    train_ppo()