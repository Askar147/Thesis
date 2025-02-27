import numpy as np
import torch
import random
import os
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# Import the UnifiedMECEnvironment
# Assuming the file is named unified_mec_environment.py
from unified_mec_environment import UnifiedMECEnvironment

# Import the GNN agent
# Assuming the file is named mec_gnn.py
from mec_gnn import MECGNNAgent, train_mec_gnn

# You would import other approaches here
# For example:
# from mec_transformer import MECTransformerAgent, train_mec_transformer
# from mec_vae import MECVAEAgent, train_mec_vae
# from mec_hybrid import MECHybridAgent, train_mec_hybrid


def seed_everything(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_comparison(num_episodes=500, max_steps=100, eval_frequency=25, seed=42):
    """
    Run a comparison of different MEC approaches using the UnifiedMECEnvironment
    
    Args:
        num_episodes: Number of episodes to train each approach
        max_steps: Maximum steps per episode
        eval_frequency: Frequency of evaluation during training
        seed: Random seed for reproducibility
        
    Returns:
        results: Dictionary of results for each approach
    """
    # Set seeds for reproducibility
    seed_everything(seed)
    
    # Create base directory for results
    base_dir = "mec_comparison_results"
    os.makedirs(base_dir, exist_ok=True)
    
    # List of approaches to compare
    approaches = [
        {
            'name': 'GNN',
            'class': MECGNNAgent,
            'train_func': train_mec_gnn,
            'checkpoint_dir': f"{base_dir}/gnn_models"
        }
        # Add other approaches here
        # {
        #     'name': 'Transformer',
        #     'class': MECTransformerAgent,
        #     'train_func': train_mec_transformer,
        #     'checkpoint_dir': f"{base_dir}/transformer_models"
        # },
        # {
        #     'name': 'VAE',
        #     'class': MECVAEAgent,
        #     'train_func': train_mec_vae,
        #     'checkpoint_dir': f"{base_dir}/vae_models"
        # },
        # {
        #     'name': 'Hybrid',
        #     'class': MECHybridAgent,
        #     'train_func': train_mec_hybrid,
        #     'checkpoint_dir': f"{base_dir}/hybrid_models"
        # }
    ]
    
    # Dictionary to store results
    results = {}
    
    # Run each approach
    for approach in approaches:
        print(f"\n{'='*50}")
        print(f"Running {approach['name']} approach")
        print(f"{'='*50}\n")
        
        # Create checkpoint directory
        os.makedirs(approach['checkpoint_dir'], exist_ok=True)
        
        # Create environment with same seed for fair comparison
        env = UnifiedMECEnvironment(
            num_edge_servers=10,
            continuous_action=False,
            history_length=20,
            difficulty='normal',
            seed=seed
        )
        
        # Calculate state and action dimensions
        state = env.reset()
        state_dim = len(env.flatten_state(state))
        action_dim = env.num_edge_servers
        
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        
        # Create agent
        agent = approach['class'](
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # Train agent
        start_time = time.time()
        trained_agent, metrics = approach['train_func'](
            env=env,
            agent=agent,
            num_episodes=num_episodes,
            max_steps=max_steps,
            eval_frequency=eval_frequency,
            save_frequency=100,
            checkpoint_dir=approach['checkpoint_dir']
        )
        training_time = time.time() - start_time
        
        # Store results
        results[approach['name']] = {
            'metrics': metrics,
            'training_time': training_time,
            'final_eval_reward': metrics['eval_rewards'][-1],
            'final_eval_latency': metrics['eval_latencies'][-1],
            'best_eval_reward': max(metrics['eval_rewards']),
            'best_eval_latency': min(metrics['eval_latencies']),
            'convergence_episode': metrics['eval_rewards'].index(max(metrics['eval_rewards'])) * eval_frequency
        }
        
        # Plot metrics for this approach
        plot_metrics(
            metrics,
            approach['name'],
            eval_frequency,
            f"{approach['checkpoint_dir']}/{approach['name']}_metrics.png"
        )
        
        # Save the results
        save_results(results, f"{base_dir}/comparison_results.txt")
        
        # Plot comparison so far
        plot_comparison(results, eval_frequency, f"{base_dir}/comparison_plot.png")
    
    return results


def plot_metrics(metrics, approach_name, eval_frequency, filename):
    """Plot training metrics for a single approach"""
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics['episode_rewards'], alpha=0.3, label='Episode Rewards')
    plt.plot(np.convolve(metrics['episode_rewards'], np.ones(20)/20, mode='valid'), 
             label='Moving Avg (20 episodes)', linewidth=2)
    
    eval_indices = [i * eval_frequency for i in range(len(metrics['eval_rewards']))]
    plt.plot(eval_indices, metrics['eval_rewards'], 'r-o', label='Evaluation Rewards', linewidth=2)
    
    plt.title(f'{approach_name} - Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot latencies
    plt.subplot(2, 2, 2)
    plt.plot(metrics['episode_latencies'], alpha=0.3, label='Episode Latencies')
    plt.plot(np.convolve(metrics['episode_latencies'], np.ones(20)/20, mode='valid'),
             label='Moving Avg (20 episodes)', linewidth=2)
    
    plt.plot(eval_indices, metrics['eval_latencies'], 'r-o', label='Evaluation Latencies', linewidth=2)
    
    plt.title(f'{approach_name} - Latencies')
    plt.xlabel('Episode')
    plt.ylabel('Latency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot losses or other metrics if available
    plt.subplot(2, 2, 3)
    if metrics.get('episode_losses'):
        plt.plot(metrics['episode_losses'], alpha=0.5)
        plt.title(f'{approach_name} - Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
    elif metrics.get('q_values'):
        plt.plot(metrics['q_values'], alpha=0.5)
        plt.title(f'{approach_name} - Q-Values')
        plt.xlabel('Episode')
        plt.ylabel('Q-Value')
        plt.grid(True, alpha=0.3)
    
    # Plot epsilon or other exploration parameter
    plt.subplot(2, 2, 4)
    if metrics.get('epsilon_values'):
        plt.plot(metrics['epsilon_values'])
        plt.title(f'{approach_name} - Exploration Rate')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_comparison(results, eval_frequency, filename):
    """Plot comparison of different approaches"""
    if len(results) < 1:
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot rewards comparison
    plt.subplot(2, 2, 1)
    for name, data in results.items():
        eval_indices = [i * eval_frequency for i in range(len(data['metrics']['eval_rewards']))]
        plt.plot(eval_indices, data['metrics']['eval_rewards'], 'o-', label=name, linewidth=2)
    
    plt.title('Evaluation Rewards Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot latencies comparison
    plt.subplot(2, 2, 2)
    for name, data in results.items():
        eval_indices = [i * eval_frequency for i in range(len(data['metrics']['eval_latencies']))]
        plt.plot(eval_indices, data['metrics']['eval_latencies'], 'o-', label=name, linewidth=2)
    
    plt.title('Evaluation Latencies Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Latency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training time comparison
    plt.subplot(2, 2, 3)
    names = list(results.keys())
    training_times = [data['training_time'] / 3600 for data in results.values()]  # Convert to hours
    
    plt.bar(names, training_times)
    plt.title('Training Time Comparison')
    plt.xlabel('Approach')
    plt.ylabel('Training Time (hours)')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot best rewards and convergence speed
    plt.subplot(2, 2, 4)
    x = np.arange(len(names))
    width = 0.35
    
    best_rewards = [data['best_eval_reward'] for data in results.values()]
    convergence_episodes = [data['convergence_episode'] for data in results.values()]
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width/2, best_rewards, width, label='Best Reward', color='skyblue')
    ax1.set_ylabel('Best Reward')
    ax1.set_ylim(min(best_rewards) * 0.9, max(best_rewards) * 1.1)
    
    bars2 = ax2.bar(x + width/2, convergence_episodes, width, label='Convergence Episode', color='salmon')
    ax2.set_ylabel('Convergence Episode')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title('Performance Metrics Comparison')
    plt.tight_layout()
    
    plt.savefig(filename)
    plt.close()


def save_results(results, filename):
    """Save comparison results to a text file"""
    with open(filename, 'w') as f:
        f.write("=== MEC Approaches Comparison Results ===\n\n")
        
        # If no results yet, just write header
        if len(results) == 0:
            f.write("No results yet.\n")
            return
        
        # Table header
        f.write(f"{'Approach':<15} {'Best Reward':<15} {'Best Latency':<15} "
                f"{'Convergence':<15} {'Training Time':<15}\n")
        f.write("-" * 75 + "\n")
        
        # Table rows
        for name, data in results.items():
            f.write(f"{name:<15} {data['best_eval_reward']:<15.2f} "
                    f"{data['best_eval_latency']:<15.4f} "
                    f"{data['convergence_episode']:<15} "
                    f"{data['training_time']/3600:<15.2f} hours\n")
        
        f.write("\n\n=== Detailed Results ===\n\n")
        
        # Detailed results for each approach
        for name, data in results.items():
            f.write(f"=== {name} ===\n")
            f.write(f"Final evaluation reward: {data['final_eval_reward']:.2f}\n")
            f.write(f"Final evaluation latency: {data['final_eval_latency']:.4f}\n")
            f.write(f"Best evaluation reward: {data['best_eval_reward']:.2f}\n")
            f.write(f"Best evaluation latency: {data['best_eval_latency']:.4f}\n")
            f.write(f"Convergence episode: {data['convergence_episode']}\n")
            f.write(f"Training time: {data['training_time']/3600:.2f} hours\n\n")


if __name__ == "__main__":
    # Run comparison with smaller number of episodes for demonstration
    results = run_comparison(
        num_episodes=300,  # Reduced for qu