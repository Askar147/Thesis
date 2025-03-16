import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import json

from vec_environment import VECEnvironment
from vec_dqn_agent import DQNAgent
from vec_dqn_train import get_state_size


def evaluate_dqn(model_path, env_config, eval_episodes=10, render=False, verbose=True):
    """Evaluate a trained DQN agent on the VEC environment"""
    # Create environment
    env = VECEnvironment(
        sumo_config=env_config['sumo_config'],
        simulation_duration=env_config['simulation_duration'],
        time_step=env_config['time_step'],
        queue_process_interval=env_config['queue_process_interval'],
        max_queue_length=env_config['max_queue_length'],
        history_length=env_config['history_length'],
        seed=env_config['seed']
    )
    
    # Calculate state size and create agent
    state_size = get_state_size(env)
    action_size = env.action_space.n
    
    if verbose:
        print(f"State size: {state_size}, Action size: {action_size}")
    
    agent = DQNAgent(state_size, action_size)
    
    # Load model
    success = agent.load_model(model_path)
    if not success:
        print(f"Failed to load model from {model_path}")
        return None
    
    # Set epsilon to a small value for some exploration during evaluation
    agent.epsilon = 0.05
    
    # Evaluation metrics
    all_rewards = []
    all_completion_rates = []
    all_rejection_rates = []
    all_drop_rates = []
    all_latencies = []
    all_steps = []
    all_node_usage = []
    all_wake_ops = []
    
    for episode in range(eval_episodes):
        state = env.reset()
        state = agent.flatten_observation(state)
        
        episode_reward = 0
        steps = 0
        wake_ops = 0
        
        while True:
            # Select action (mostly exploitation)
            action = agent.select_action(state)
            
            # Track wake up operations
            if action == env.nodes_per_bs:
                wake_ops += 1
            
            # Take action
            next_state, reward, done, info = env.step(action)
            next_state = agent.flatten_observation(next_state)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Optional rendering
            if render and steps % 10 == 0:
                env.render()
            
            if done:
                break
        
        # Record metrics for this episode
        all_rewards.append(episode_reward)
        all_steps.append(steps)
        all_wake_ops.append(wake_ops)
        
        # Record environment metrics
        all_completion_rates.append(info.get('task_completion_rate', 0))
        all_rejection_rates.append(info.get('task_rejection_rate', 0))
        all_drop_rates.append(info.get('task_drop_rate', 0))
        all_latencies.append(info.get('avg_latency', 0))
        
        # Calculate node usage - the average percentage of active nodes
        # This is just an example - you might need to adjust based on your env
        node_usage = np.mean([sum(bs_instance.nodes[i].active for i in range(env.nodes_per_bs)) / env.nodes_per_bs 
                              for bs_instance in env.base_station_instances.values()])
        all_node_usage.append(node_usage)
        
        if verbose:
            print(f"Episode {episode+1} - " +
                 f"Reward: {episode_reward:.2f}, " +
                 f"Steps: {steps}, " +
                 f"Wake ops: {wake_ops}, " +
                 f"Completion rate: {info.get('task_completion_rate', 0):.3f}, " +
                 f"Avg latency: {info.get('avg_latency', 0):.3f}, " +
                 f"Node usage: {node_usage:.3f}")
    
    # Compile evaluation results
    eval_results = {
        'model_path': model_path,
        'environment': env_config,
        'episodes': eval_episodes,
        'metrics': {
            'rewards': {
                'mean': float(np.mean(all_rewards)),
                'std': float(np.std(all_rewards)),
                'min': float(np.min(all_rewards)),
                'max': float(np.max(all_rewards)),
                'values': [float(r) for r in all_rewards]
            },
            'completion_rates': {
                'mean': float(np.mean(all_completion_rates)),
                'std': float(np.std(all_completion_rates)),
                'values': [float(r) for r in all_completion_rates]
            },
            'rejection_rates': {
                'mean': float(np.mean(all_rejection_rates)),
                'std': float(np.std(all_rejection_rates)),
                'values': [float(r) for r in all_rejection_rates]
            },
            'drop_rates': {
                'mean': float(np.mean(all_drop_rates)),
                'std': float(np.std(all_drop_rates)),
                'values': [float(r) for r in all_drop_rates]
            },
            'latencies': {
                'mean': float(np.mean(all_latencies)),
                'std': float(np.std(all_latencies)),
                'values': [float(l) for l in all_latencies]
            },
            'steps': {
                'mean': float(np.mean(all_steps)),
                'std': float(np.std(all_steps)),
                'values': [int(s) for s in all_steps]
            },
            'node_usage': {
                'mean': float(np.mean(all_node_usage)),
                'std': float(np.std(all_node_usage)),
                'values': [float(n) for n in all_node_usage]
            },
            'wake_operations': {
                'mean': float(np.mean(all_wake_ops)),
                'std': float(np.std(all_wake_ops)),
                'values': [int(w) for w in all_wake_ops]
            }
        }
    }
    
    if verbose:
        print("\nEvaluation Summary:")
        print(f"Average reward: {eval_results['metrics']['rewards']['mean']:.2f}")
        print(f"Average completion rate: {eval_results['metrics']['completion_rates']['mean']:.3f}")
        print(f"Average latency: {eval_results['metrics']['latencies']['mean']:.3f}")
        print(f"Average node usage: {eval_results['metrics']['node_usage']['mean']:.3f}")
        print(f"Average wake operations: {eval_results['metrics']['wake_operations']['mean']:.1f}")
    
    return eval_results


def plot_evaluation_results(eval_results, save_dir=None):
    """Plot evaluation results"""
    plt.figure(figsize=(18, 12))
    
    # Plot rewards distribution
    plt.subplot(2, 3, 1)
    plt.hist(eval_results['metrics']['rewards']['values'], bins=10, alpha=0.7)
    plt.axvline(eval_results['metrics']['rewards']['mean'], color='r', linestyle='dashed', linewidth=1)
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title(f"Rewards (Mean: {eval_results['metrics']['rewards']['mean']:.2f})")
    plt.grid(True)
    
    # Plot completion rates
    plt.subplot(2, 3, 2)
    plt.hist(eval_results['metrics']['completion_rates']['values'], bins=10, alpha=0.7)
    plt.axvline(eval_results['metrics']['completion_rates']['mean'], color='r', linestyle='dashed', linewidth=1)
    plt.xlabel('Completion Rate')
    plt.ylabel('Frequency')
    plt.title(f"Completion Rates (Mean: {eval_results['metrics']['completion_rates']['mean']:.3f})")
    plt.grid(True)
    
    # Plot latencies
    plt.subplot(2, 3, 3)
    plt.hist(eval_results['metrics']['latencies']['values'], bins=10, alpha=0.7)
    plt.axvline(eval_results['metrics']['latencies']['mean'], color='r', linestyle='dashed', linewidth=1)
    plt.xlabel('Average Latency (s)')
    plt.ylabel('Frequency')
    plt.title(f"Latencies (Mean: {eval_results['metrics']['latencies']['mean']:.3f}s)")
    plt.grid(True)
    
    # Plot node usage
    plt.subplot(2, 3, 4)
    plt.hist(eval_results['metrics']['node_usage']['values'], bins=10, alpha=0.7)
    plt.axvline(eval_results['metrics']['node_usage']['mean'], color='r', linestyle='dashed', linewidth=1)
    plt.xlabel('Node Usage (%)')
    plt.ylabel('Frequency')
    plt.title(f"Node Usage (Mean: {eval_results['metrics']['node_usage']['mean']:.3f})")
    plt.grid(True)
    
    # Plot wake operations
    plt.subplot(2, 3, 5)
    plt.hist(eval_results['metrics']['wake_operations']['values'], bins=10, alpha=0.7)
    plt.axvline(eval_results['metrics']['wake_operations']['mean'], color='r', linestyle='dashed', linewidth=1)
    plt.xlabel('Wake Operations')
    plt.ylabel('Frequency')
    plt.title(f"Wake Operations (Mean: {eval_results['metrics']['wake_operations']['mean']:.1f})")
    plt.grid(True)
    
    # Plot rejection + drop rates
    plt.subplot(2, 3, 6)
    rejection_rates = eval_results['metrics']['rejection_rates']['values']
    drop_rates = eval_results['metrics']['drop_rates']['values']
    
    plt.hist([rejection_rates, drop_rates], label=['Rejection Rate', 'Drop Rate'], alpha=0.7)
    plt.axvline(eval_results['metrics']['rejection_rates']['mean'], color='r', linestyle='dashed', linewidth=1)
    plt.axvline(eval_results['metrics']['drop_rates']['mean'], color='g', linestyle='dashed', linewidth=1)
    plt.xlabel('Rate')
    plt.ylabel('Frequency')
    plt.title(f"Rejection (Mean: {eval_results['metrics']['rejection_rates']['mean']:.3f}) & Drop (Mean: {eval_results['metrics']['drop_rates']['mean']:.3f})")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, 'evaluation_results.png'), dpi=300)
    
    plt.show()


def compare_models(models, env_config, eval_episodes=10, save_dir=None):
    """Compare multiple trained models"""
    results = []
    
    for model_info in models:
        model_path = model_info['path']
        model_name = model_info.get('name', os.path.basename(model_path))
        
        print(f"\nEvaluating model: {model_name}")
        eval_result = evaluate_dqn(model_path, env_config, eval_episodes, verbose=False)
        
        if eval_result:
            eval_result['name'] = model_name
            results.append(eval_result)
    
    if not results:
        print("No models were successfully evaluated.")
        return
    
    # Create comparison plots
    plt.figure(figsize=(18, 12))
    
    # Metrics to compare
    metrics = {
        'rewards': 'Episode Reward',
        'completion_rates': 'Task Completion Rate',
        'latencies': 'Average Latency (s)',
        'node_usage': 'Node Usage',
        'wake_operations': 'Wake Operations',
        'rejection_rates': 'Task Rejection Rate'
    }
    
    # Plot each metric
    for i, (metric_key, metric_name) in enumerate(metrics.items(), 1):
        plt.subplot(2, 3, i)
        
        # Extract data for this metric
        data = []
        labels = []
        for result in results:
            data.append(result['metrics'][metric_key]['mean'])
            labels.append(result['name'])
        
        # Create bar chart
        bars = plt.bar(labels, data, alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(data),
                    f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        plt.xlabel('Model')
        plt.ylabel(metric_name)
        plt.title(f'Comparison of {metric_name}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y')
    
    plt.tight_layout()
    
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300)
        
        # Also save comparison data as JSON
        comparison_data = {
            'models': [r['name'] for r in results],
            'metrics': {}
        }
        
        for metric_key in metrics.keys():
            comparison_data['metrics'][metric_key] = {
                r['name']: {
                    'mean': r['metrics'][metric_key]['mean'],
                    'std': r['metrics'][metric_key]['std']
                } for r in results
            }
        
        with open(os.path.join(save_dir, 'model_comparison.json'), 'w') as f:
            json.dump(comparison_data, f, indent=4)
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DQN agent for VEC task offloading')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--sumo_config', type=str, default='astana.sumocfg', 
                        help='Path to SUMO configuration file')
    parser.add_argument('--simulation_duration', type=int, default=300,
                        help='Total simulation duration (seconds)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed (different from training)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Environment configuration - similar to training but with different seed
    env_config = {
        'sumo_config': args.sumo_config,
        'simulation_duration': args.simulation_duration,
        'time_step': 1,
        'queue_process_interval': 5,
        'max_queue_length': 50,
        'history_length': 10,
        'seed': args.seed
    }
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate the model
    results = evaluate_dqn(args.model_path, env_config, args.episodes, args.render)
    
    if results:
        # Save evaluation results
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Plot results
        plot_evaluation_results(results, save_dir=output_dir)
        
        print(f"\nEvaluation results saved to {output_dir}")