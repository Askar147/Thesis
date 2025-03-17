import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Define paths directly based on your directory structure
dqn_metrics_path = "results/run_train_20250317_004054/run_20250317_004057/metrics.json"
dqn_model_path = "results/run_train_20250317_004054/run_20250317_004057/final_model.pth"

transformer_metrics_path = "transformer_results\\run_train_20250317_005431\\transformer_metrics_20250317_035219.json"
transformer_model_path = "transformer_results/run_20250317_005431/transformer_model_final.pt"

# Create output directory
output_dir = "custom_comparison_results"
os.makedirs(output_dir, exist_ok=True)

# Load DQN metrics
print(f"Loading DQN metrics from: {dqn_metrics_path}")
if os.path.exists(dqn_metrics_path):
    with open(dqn_metrics_path, 'r') as f:
        dqn_metrics = json.load(f)
    print("Successfully loaded DQN metrics")
else:
    print(f"Error: DQN metrics file not found at {dqn_metrics_path}")
    dqn_metrics = {}

# Load Transformer metrics
print(f"Loading Transformer metrics from: {transformer_metrics_path}")
if os.path.exists(transformer_metrics_path):
    with open(transformer_metrics_path, 'r') as f:
        transformer_metrics = json.load(f)
    print("Successfully loaded Transformer metrics")
else:
    print(f"Error: Transformer metrics file not found at {transformer_metrics_path}")
    transformer_metrics = {}

# Check if we have metrics to compare
if not dqn_metrics or not transformer_metrics:
    print("Not enough data to create comparison visualizations")
    exit()

# Create comparison visualizations
plt.figure(figsize=(20, 15))

# Plot rewards
plt.subplot(2, 2, 1)
if 'rewards' in dqn_metrics and 'rewards' in transformer_metrics:
    plt.plot(dqn_metrics['rewards'], alpha=0.3, label='DQN Rewards')
    plt.plot(transformer_metrics['rewards'], alpha=0.3, label='Transformer Rewards')
    
    # Calculate and plot moving averages
    window_size = 20
    if len(dqn_metrics['rewards']) > window_size:
        dqn_moving_avg = np.convolve(dqn_metrics['rewards'], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
        x_avg = np.arange(window_size-1, len(dqn_metrics['rewards']))
        plt.plot(x_avg, dqn_moving_avg, 'r-', label='DQN Moving Avg', linewidth=2)
        
    if len(transformer_metrics['rewards']) > window_size:
        transformer_moving_avg = np.convolve(transformer_metrics['rewards'], 
                                          np.ones(window_size)/window_size, 
                                          mode='valid')
        x_avg = np.arange(window_size-1, len(transformer_metrics['rewards']))
        plt.plot(x_avg, transformer_moving_avg, 'b-', label='Transformer Moving Avg', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Reward Comparison')
    plt.grid(True)

# Plot task completion rates
plt.subplot(2, 2, 2)
dqn_completion_key = 'task_completion_rates' if 'task_completion_rates' in dqn_metrics else 'completion_rates' 
transformer_completion_key = 'task_completion_rates' if 'task_completion_rates' in transformer_metrics else 'completion_rates'

if dqn_completion_key in dqn_metrics and transformer_completion_key in transformer_metrics:
    plt.plot(dqn_metrics[dqn_completion_key], alpha=0.3, label='DQN')
    plt.plot(transformer_metrics[transformer_completion_key], alpha=0.3, label='Transformer')
    
    # Calculate and plot moving averages
    if len(dqn_metrics[dqn_completion_key]) > window_size:
        dqn_moving_avg = np.convolve(dqn_metrics[dqn_completion_key], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
        x_avg = np.arange(window_size-1, len(dqn_metrics[dqn_completion_key]))
        plt.plot(x_avg, dqn_moving_avg, 'r-', label='DQN Moving Avg', linewidth=2)
        
    if len(transformer_metrics[transformer_completion_key]) > window_size:
        transformer_moving_avg = np.convolve(transformer_metrics[transformer_completion_key], 
                                          np.ones(window_size)/window_size, 
                                          mode='valid')
        x_avg = np.arange(window_size-1, len(transformer_metrics[transformer_completion_key]))
        plt.plot(x_avg, transformer_moving_avg, 'b-', label='Transformer Moving Avg', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Completion Rate')
    plt.legend()
    plt.title('Task Completion Rate Comparison')
    plt.grid(True)

# Plot energy consumption
plt.subplot(2, 2, 3)
dqn_energy_key = 'energy_consumptions' if 'energy_consumptions' in dqn_metrics else 'avg_energy_consumption' 
transformer_energy_key = 'energy_consumptions' if 'energy_consumptions' in transformer_metrics else 'avg_energy_consumption'

if dqn_energy_key in dqn_metrics and transformer_energy_key in transformer_metrics:
    plt.plot(dqn_metrics[dqn_energy_key], alpha=0.3, label='DQN')
    plt.plot(transformer_metrics[transformer_energy_key], alpha=0.3, label='Transformer')
    
    # Calculate and plot moving averages
    if len(dqn_metrics[dqn_energy_key]) > window_size:
        dqn_moving_avg = np.convolve(dqn_metrics[dqn_energy_key], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
        x_avg = np.arange(window_size-1, len(dqn_metrics[dqn_energy_key]))
        plt.plot(x_avg, dqn_moving_avg, 'r-', label='DQN Moving Avg', linewidth=2)
        
    if len(transformer_metrics[transformer_energy_key]) > window_size:
        transformer_moving_avg = np.convolve(transformer_metrics[transformer_energy_key], 
                                          np.ones(window_size)/window_size, 
                                          mode='valid')
        x_avg = np.arange(window_size-1, len(transformer_metrics[transformer_energy_key]))
        plt.plot(x_avg, transformer_moving_avg, 'b-', label='Transformer Moving Avg', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.title('Energy Consumption Comparison')
    plt.grid(True)

# Plot latency
plt.subplot(2, 2, 4)
dqn_latency_key = 'avg_latencies' if 'avg_latencies' in dqn_metrics else 'latencies' 
transformer_latency_key = 'latencies' if 'latencies' in transformer_metrics else 'avg_latencies'

if dqn_latency_key in dqn_metrics and transformer_latency_key in transformer_metrics:
    plt.plot(dqn_metrics[dqn_latency_key], alpha=0.3, label='DQN')
    plt.plot(transformer_metrics[transformer_latency_key], alpha=0.3, label='Transformer')
    
    # Calculate and plot moving averages
    if len(dqn_metrics[dqn_latency_key]) > window_size:
        dqn_moving_avg = np.convolve(dqn_metrics[dqn_latency_key], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
        x_avg = np.arange(window_size-1, len(dqn_metrics[dqn_latency_key]))
        plt.plot(x_avg, dqn_moving_avg, 'r-', label='DQN Moving Avg', linewidth=2)
        
    if len(transformer_metrics[transformer_latency_key]) > window_size:
        transformer_moving_avg = np.convolve(transformer_metrics[transformer_latency_key], 
                                          np.ones(window_size)/window_size, 
                                          mode='valid')
        x_avg = np.arange(window_size-1, len(transformer_metrics[transformer_latency_key]))
        plt.plot(x_avg, transformer_moving_avg, 'b-', label='Transformer Moving Avg', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Latency (s)')
    plt.legend()
    plt.title('Latency Comparison')
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)

# Create summary table
print("\nPerformance Comparison Summary:")
print("=" * 50)
print(f"{'Metric':<20} {'DQN':<15} {'Transformer':<15} {'Difference':<15} {'Improvement %':<15}")
print("-" * 80)

# Compare final average metrics
metrics_to_compare = [
    {'name': 'Avg Reward', 'dqn': 'rewards', 'transformer': 'rewards', 'better': 'higher'},
    {'name': 'Completion Rate', 'dqn': dqn_completion_key, 'transformer': transformer_completion_key, 'better': 'higher'},
    {'name': 'Latency', 'dqn': dqn_latency_key, 'transformer': transformer_latency_key, 'better': 'lower'},
    {'name': 'Energy', 'dqn': dqn_energy_key, 'transformer': transformer_energy_key, 'better': 'lower'}
]

summary_data = {}
for metric in metrics_to_compare:
    if metric['dqn'] in dqn_metrics and metric['transformer'] in transformer_metrics:
        # Calculate averages of last 100 episodes (or all if less than 100)
        dqn_values = dqn_metrics[metric['dqn']]
        transformer_values = transformer_metrics[metric['transformer']]
        
        dqn_avg = np.mean(dqn_values[-100:] if len(dqn_values) >= 100 else dqn_values)
        transformer_avg = np.mean(transformer_values[-100:] if len(transformer_values) >= 100 else transformer_values)
        
        # Calculate difference and improvement
        diff = transformer_avg - dqn_avg
        if metric['better'] == 'lower':
            improvement = (dqn_avg - transformer_avg) / dqn_avg * 100 if dqn_avg != 0 else float('inf')
            better = improvement > 0
        else:  # higher is better
            improvement = (transformer_avg - dqn_avg) / dqn_avg * 100 if dqn_avg != 0 else float('inf')
            better = improvement > 0
            
        # Store for JSON - Convert numpy types to native Python types
        summary_data[metric['name']] = {
            'DQN': float(dqn_avg),
            'Transformer': float(transformer_avg),
            'Difference': float(diff),
            'Improvement_percent': float(improvement),
            'Transformer_is_better': bool(better)  # Convert numpy bool to Python bool
        }
        
        # Print
        print(f"{metric['name']:<20} {dqn_avg:<15.4f} {transformer_avg:<15.4f} {diff:<15.4f} {improvement:<15.2f}%")

# Save summary as JSON
with open(os.path.join(output_dir, 'comparison_summary.json'), 'w') as f:
    json.dump(summary_data, f, indent=4)

print("\nComparison visualizations and summary saved to:", output_dir)