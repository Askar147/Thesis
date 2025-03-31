#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from vec_environment import VECEnvironment
from ff_dqn_agent import DQNAgent
from te_dqn_agent import VECTransformerAgent, flatten_observation as te_flatten_observation

class ModelAnalyzer:
    """Analyze and visualize model decision-making processes"""
    
    def __init__(self, ff_model_path, te_model_path, env_config=None):
        """Initialize with paths to FF-DQN and TE-DQN models"""
        self.ff_model_path = ff_model_path
        self.te_model_path = te_model_path
        
        # Default environment config if none provided
        if env_config is None:
            self.env_config = {
                'sumo_config': 'astana.sumocfg',
                'simulation_duration': 300,
                'time_step': 1,
                'queue_process_interval': 5,
                'max_queue_length': 50,
                'history_length': 10,
                'energy_csv_path': 'merged_dag1.csv',
                'energy_weight': 0.5,
                'min_tasks_per_step': 10,
                'max_tasks_per_step': 20,
                'task_generation_probability': 1.0,
                'seed': 42
            }
        else:
            self.env_config = env_config
        
        # Create environment
        self.env = VECEnvironment(**self.env_config)
        
        # Get state and action sizes
        obs = self.env.reset()
        temp_agent = DQNAgent(0, self.env.action_space.n)
        flattened = temp_agent.flatten_observation(obs)
        self.state_size = len(flattened)
        self.action_size = self.env.action_space.n
        
        # Load models
        self.ff_agent = self._load_ff_model()
        self.te_agent = self._load_te_model()
    
    def _load_ff_model(self):
        """Load FF-DQN model"""
        agent = DQNAgent(self.state_size, self.action_size)
        if os.path.exists(self.ff_model_path):
            agent.load_model(self.ff_model_path)
            print(f"Loaded FF-DQN model from {self.ff_model_path}")
            return agent
        else:
            print(f"Warning: FF-DQN model not found at {self.ff_model_path}")
            return None
    
    def _load_te_model(self):
        """Load TE-DQN model"""
        agent = VECTransformerAgent(self.state_size, self.action_size)
        if os.path.exists(self.te_model_path):
            agent.load_model(self.te_model_path)
            print(f"Loaded TE-DQN model from {self.te_model_path}")
            return agent
        else:
            print(f"Warning: TE-DQN model not found at {self.te_model_path}")
            return None
    
    def analyze_decision_making(self, num_steps=100):
        """Analyze and compare decision-making processes of both models"""
        if not self.ff_agent or not self.te_agent:
            print("Error: One or both models not loaded")
            return
        
        # Reset environment
        obs = self.env.reset()
        
        # Clear TE-DQN state history
        self.te_agent.state_history.clear()
        
        # Track decisions
        decisions = {
            'ff': {
                'actions': [],
                'q_values': [],
                'rewards': [],
                'wake_actions': 0
            },
            'te': {
                'actions': [],
                'q_values': [],
                'rewards': [],
                'wake_actions': 0
            }
        }
        
        # Environment metrics
        env_metrics = {
            'queue_lengths': [],
            'active_nodes': [],
            'task_completion_rates': [],
            'energy_consumptions': []
        }
        
        # Run for specified steps
        for step in range(num_steps):
            # Get FF-DQN action and q-values
            ff_state = self.ff_agent.flatten_observation(obs)
            
            with torch.no_grad():
                ff_state_tensor = torch.FloatTensor(ff_state).unsqueeze(0).to(self.ff_agent.device)
                ff_q_values = self.ff_agent.q_network(ff_state_tensor).cpu().numpy()[0]
            
            ff_action = ff_q_values.argmax()
            decisions['ff']['actions'].append(int(ff_action))
            decisions['ff']['q_values'].append(ff_q_values.tolist())
            
            if ff_action == self.action_size - 1:  # Wake action
                decisions['ff']['wake_actions'] += 1
            
            # Get TE-DQN action and q-values
            te_state = te_flatten_observation(obs)
            self.te_agent.state_history.append(te_state)
            
            # Pad history if needed
            if len(self.te_agent.state_history) < self.te_agent.seq_length:
                padding = [te_state] * (self.te_agent.seq_length - len(self.te_agent.state_history))
                seq_states = padding + list(self.te_agent.state_history)
            else:
                seq_states = list(self.te_agent.state_history)
            
            seq_tensor = torch.FloatTensor(np.array(seq_states)).to(self.te_agent.device)
            
            with torch.no_grad():
                te_q_values, _ = self.te_agent.policy_net(seq_tensor)
                te_q_values = te_q_values.cpu().numpy()[0]
            
            te_action = te_q_values.argmax()
            decisions['te']['actions'].append(int(te_action))
            decisions['te']['q_values'].append(te_q_values.tolist())
            
            if te_action == self.action_size - 1:  # Wake action
                decisions['te']['wake_actions'] += 1
            
            # Use FF-DQN action for environment step (arbitrary choice)
            next_obs, reward, done, info = self.env.step(ff_action)
            decisions['ff']['rewards'].append(float(reward))
            
            # Now step with TE-DQN action in a copy of the observation (hypothetical)
            # Note: This is just for analysis, not accurate simulation
            te_reward = self._estimate_reward(obs, te_action)
            decisions['te']['rewards'].append(float(te_reward))
            
            # Track environment metrics
            total_queue = sum(len(bs.queue) for bs in self.env.base_station_instances.values())
            env_metrics['queue_lengths'].append(total_queue)
            
            active_nodes = sum(sum(1 for node in bs.nodes if node.active) 
                              for bs in self.env.base_station_instances.values())
            total_nodes = sum(len(bs.nodes) for bs in self.env.base_station_instances.values())
            env_metrics['active_nodes'].append(active_nodes / total_nodes)
            
            env_metrics['task_completion_rates'].append(info.get('task_completion_rate', 0))
            env_metrics['energy_consumptions'].append(info.get('avg_energy_consumption', 0))
            
            obs = next_obs
            
            if done:
                break
        
        # Compute decision agreement
        agreement = sum(1 for i in range(len(decisions['ff']['actions'])) 
                       if decisions['ff']['actions'][i] == decisions['te']['actions'][i])
        agreement_rate = agreement / len(decisions['ff']['actions']) if decisions['ff']['actions'] else 0
        
        # Create visualizations
        self._visualize_decisions(decisions, env_metrics, agreement_rate)
        
        return {
            'decisions': decisions,
            'env_metrics': env_metrics,
            'steps_completed': min(step + 1, num_steps),
            'agreement_rate': agreement_rate
        }
    
    def _estimate_reward(self, obs, action):
        """Rough estimation of reward for TE-DQN action without actually stepping environment"""
        # This is a simplified estimation - not accurate for true comparison
        if action == self.action_size - 1:  # Wake action
            # Usually a negative reward for energy cost
            return -0.5
        
        # For other actions, estimate based on node load
        bs = self.env.current_bs
        if bs is None:
            return 0
        
        node_loads = [len(node.active_tasks) / node.max_concurrent_tasks 
                     for node in bs.nodes]
        
        # Lower load means higher potential reward
        if action < len(node_loads) and action >= 0:
            # Better reward for less loaded nodes
            load_factor = 1.0 - node_loads[action]
            return 0.5 * load_factor
        else:
            return 0
    
    def _visualize_decisions(self, decisions, env_metrics, agreement_rate):
        """Create visualizations of model decisions and environmental factors"""
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Add title with agreement rate
        fig.suptitle(f"Model Decision Analysis (Agreement Rate: {agreement_rate:.2f})", 
                   fontsize=16)
        
        # 1. Action choices over time
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(decisions['ff']['actions'], 'b-', label='FF-DQN Actions')
        ax1.plot(decisions['te']['actions'], 'r-', label='TE-DQN Actions')
        ax1.set_title('Action Choices Over Time')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Action ID')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Rewards
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(decisions['ff']['rewards'], 'b-', label='FF-DQN Rewards')
        ax2.plot(decisions['te']['rewards'], 'r-', label='TE-DQN Rewards (est.)')
        # Calculate cumulative rewards
        ff_cum_rewards = np.cumsum(decisions['ff']['rewards'])
        te_cum_rewards = np.cumsum(decisions['te']['rewards'])
        ax2.plot(ff_cum_rewards, 'b--', alpha=0.5, label='FF-DQN Cumulative')
        ax2.plot(te_cum_rewards, 'r--', alpha=0.5, label='TE-DQN Cumulative')
        ax2.set_title('Rewards Over Time')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Wake-up decisions (vertical lines)
        ax3 = fig.add_subplot(3, 2, 3)
        # Plot vertical lines at wake-up decisions
        for i, action in enumerate(decisions['ff']['actions']):
            if action == self.action_size - 1:
                ax3.axvline(x=i, color='blue', alpha=0.5, linestyle='--')
        
        for i, action in enumerate(decisions['te']['actions']):
            if action == self.action_size - 1:
                ax3.axvline(x=i, color='red', alpha=0.5, linestyle='--')
        
        # Plot queue length as background context
        ax3.plot(env_metrics['queue_lengths'], 'g-', label='Queue Length')
        ax3.set_title('Wake-up Decisions & Queue Length')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Queue Length')
        # Add custom legend for wake-up decisions
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='blue', linestyle='--', alpha=0.5),
                       Line2D([0], [0], color='red', linestyle='--', alpha=0.5),
                       Line2D([0], [0], color='green')]
        ax3.legend(custom_lines, ['FF-DQN Wake', 'TE-DQN Wake', 'Queue Length'])
        ax3.grid(True)
        
        # 4. Q-value heatmap comparison for a sample step
        sample_step = min(10, len(decisions['ff']['q_values']) - 1)
        ax4 = fig.add_subplot(3, 2, 4)
        
        if sample_step >= 0:
            # Get q-values for sample step
            ff_q = np.array(decisions['ff']['q_values'][sample_step])
            te_q = np.array(decisions['te']['q_values'][sample_step])
            
            # Normalize for better visualization
            ff_q = (ff_q - ff_q.min()) / (ff_q.max() - ff_q.min() + 1e-8)
            te_q = (te_q - te_q.min()) / (te_q.max() - te_q.min() + 1e-8)
            
            # Plot as histogram
            x = np.arange(len(ff_q))
            width = 0.35
            ax4.bar(x - width/2, ff_q, width, label='FF-DQN Q-Values')
            ax4.bar(x + width/2, te_q, width, label='TE-DQN Q-Values')
            
            # Mark chosen actions
            ff_action = decisions['ff']['actions'][sample_step]
            te_action = decisions['te']['actions'][sample_step]
            ax4.plot(ff_action - width/2, ff_q[ff_action], 'bo', markersize=10)
            ax4.plot(te_action + width/2, te_q[te_action], 'ro', markersize=10)
            
            ax4.set_title(f'Q-Value Comparison (Step {sample_step})')
            ax4.set_xlabel('Action')
            ax4.set_ylabel('Normalized Q-Value')
            ax4.legend()
            ax4.grid(True)
        
        # 5. Environmental metrics
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.plot(env_metrics['active_nodes'], 'g-', label='Active Node Ratio')
        ax5.plot(env_metrics['task_completion_rates'], 'm-', label='Task Completion Rate')
        ax5.set_title('Environmental Metrics')
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Value')
        ax5.legend()
        ax5.grid(True)
        
        # 6. Energy consumption
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.plot(env_metrics['energy_consumptions'], 'k-', label='Energy Consumption')
        ax6.set_title('Energy Consumption')
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Energy (J)')
        ax6.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        plt.savefig('model_decision_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Decision analysis visualization saved to model_decision_analysis.png")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze model behavior")
    parser.add_argument('--ff_model', type=str, required=True, help='Path to FF-DQN model')
    parser.add_argument('--te_model', type=str, required=True, help='Path to TE-DQN model')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to analyze')
    
    args = parser.parse_args()
    
    analyzer = ModelAnalyzer(args.ff_model, args.te_model)
    results = analyzer.analyze_decision_making(args.steps)
    
    print(f"Analysis complete. Agreement rate: {results['agreement_rate']:.2f}")
    print(f"FF-DQN wake decisions: {results['decisions']['ff']['wake_actions']}")
    print(f"TE-DQN wake decisions: {results['decisions']['te']['wake_actions']}")