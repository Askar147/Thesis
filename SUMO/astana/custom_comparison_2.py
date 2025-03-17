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
output_dir = "enhanced_comparison_results"
os.makedirs(output_dir, exist_ok=True)

# Load DQN metrics
print(f"Loading DQN metrics from: {dqn_metrics_path}")
if os.path.exists(dqn_metrics_path):
    with open(dqn_metrics_path, 'r') as f:
        dqn_metrics = json.load(f)
    print("Successfully loaded DQN metrics")
    # Print available keys for debugging
    print(f"DQN metrics keys: {list(dqn_metrics.keys())}")
else:
    print(f"Error: DQN metrics file not found at {dqn_metrics_path}")
    dqn_metrics = {}

# Load Transformer metrics
print(f"Loading Transformer metrics from: {transformer_metrics_path}")
if os.path.exists(transformer_metrics_path):
    with open(transformer_metrics_path, 'r') as f:
        transformer_metrics = json.load(f)
    print("Successfully loaded Transformer metrics")
    # Print available keys for debugging
    print(f"Transformer metrics keys: {list(transformer_metrics.keys())}")
else:
    print(f"Error: Transformer metrics file not found at {transformer_metrics_path}")
    transformer_metrics = {}

# Check if we have metrics to compare
if not dqn_metrics or not transformer_metrics:
    print("Not enough data to create comparison visualizations")
    exit()

# Normalize lengths if needed
def get_equal_length_data(dqn_data, transformer_data, max_len=1000):
    """Make sure both datasets are the same length for easier plotting"""
    dqn_len = min(len(dqn_data), max_len)
    transformer_len = min(len(transformer_data), max_len)
    
    # Get the minimum length of both datasets
    min_len = min(dqn_len, transformer_len)
    
    # Take the last min_len elements from each dataset
    dqn_data = dqn_data[-min_len:]
    transformer_data = transformer_data[-min_len:]
    
    return dqn_data, transformer_data, min_len

# Create multiplot comparison visualizations
plt.figure(figsize=(18, 14))

# Plot rewards
plt.subplot(3, 2, 1)
if 'rewards' in dqn_metrics and 'rewards' in transformer_metrics:
    # Ensure equal length for proper comparison
    dqn_rewards, trans_rewards, length = get_equal_length_data(
        dqn_metrics['rewards'], transformer_metrics['rewards']
    )
    
    episodes = range(length)
    plt.plot(episodes, dqn_rewards, 'r-', alpha=0.7, label='DQN')
    plt.plot(episodes, trans_rewards, 'b-', alpha=0.7, label='Transformer')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Reward Comparison')
    plt.grid(True)

# Plot task completion rates
plt.subplot(3, 2, 2)
dqn_completion_key = next((k for k in dqn_metrics.keys() if 'completion' in k.lower()), None)
transformer_completion_key = next((k for k in transformer_metrics.keys() if 'completion' in k.lower()), None)

if dqn_completion_key and transformer_completion_key:
    # Ensure equal length for proper comparison
    dqn_completion, trans_completion, length = get_equal_length_data(
        dqn_metrics[dqn_completion_key], transformer_metrics[transformer_completion_key]
    )
    
    episodes = range(length)
    plt.plot(episodes, dqn_completion, 'r-', alpha=0.7, label='DQN')
    plt.plot(episodes, trans_completion, 'b-', alpha=0.7, label='Transformer')
    
    plt.xlabel('Episode')
    plt.ylabel('Completion Rate')
    plt.legend()
    plt.title('Task Completion Rate Comparison')
    plt.grid(True)

# Plot energy consumption
plt.subplot(3, 2, 3)
dqn_energy_key = next((k for k in dqn_metrics.keys() if 'energy' in k.lower()), None)
transformer_energy_key = next((k for k in transformer_metrics.keys() if 'energy' in k.lower()), None)

if dqn_energy_key and transformer_energy_key:
    # Ensure equal length for proper comparison
    dqn_energy, trans_energy, length = get_equal_length_data(
        dqn_metrics[dqn_energy_key], transformer_metrics[transformer_energy_key]
    )
    
    episodes = range(length)
    plt.plot(episodes, dqn_energy, 'r-', alpha=0.7, label='DQN')
    plt.plot(episodes, trans_energy, 'b-', alpha=0.7, label='Transformer')
    
    plt.xlabel('Episode')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.title('Energy Consumption Comparison')
    plt.grid(True)

# Plot latency
plt.subplot(3, 2, 4)
dqn_latency_key = next((k for k in dqn_metrics.keys() if 'latenc' in k.lower()), None)
transformer_latency_key = next((k for k in transformer_metrics.keys() if 'latenc' in k.lower()), None)

if dqn_latency_key and transformer_latency_key:
    # Ensure equal length for proper comparison
    dqn_latency, trans_latency, length = get_equal_length_data(
        dqn_metrics[dqn_latency_key], transformer_metrics[transformer_latency_key]
    )
    
    episodes = range(length)
    plt.plot(episodes, dqn_latency, 'r-', alpha=0.7, label='DQN')
    plt.plot(episodes, trans_latency, 'b-', alpha=0.7, label='Transformer')
    
    plt.xlabel('Episode')
    plt.ylabel('Latency (s)')
    plt.legend()
    plt.title('Latency Comparison')
    plt.grid(True)

# Plot node usage/utilization if available
plt.subplot(3, 2, 5)
dqn_node_key = next((k for k in dqn_metrics.keys() if 'node' in k.lower() or 'util' in k.lower()), None)
transformer_node_key = next((k for k in transformer_metrics.keys() if 'node' in k.lower() or 'util' in k.lower()), None)

if dqn_node_key and transformer_node_key:
    # Ensure equal length for proper comparison
    dqn_node, trans_node, length = get_equal_length_data(
        dqn_metrics[dqn_node_key], transformer_metrics[transformer_node_key]
    )
    
    episodes = range(length)
    plt.plot(episodes, dqn_node, 'r-', alpha=0.7, label='DQN')
    plt.plot(episodes, trans_node, 'b-', alpha=0.7, label='Transformer')
    
    plt.xlabel('Episode')
    plt.ylabel('Node Utilization')
    plt.legend()
    plt.title('Node Utilization Comparison')
    plt.grid(True)

# Plot training loss if available
plt.subplot(3, 2, 6)
dqn_loss_key = next((k for k in dqn_metrics.keys() if 'loss' in k.lower()), None)
transformer_loss_key = next((k for k in transformer_metrics.keys() if 'loss' in k.lower()), None)

if dqn_loss_key and transformer_loss_key:
    # Ensure equal length for proper comparison
    dqn_loss, trans_loss, length = get_equal_length_data(
        dqn_metrics[dqn_loss_key], transformer_metrics[transformer_loss_key], max_len=5000
    )
    
    steps = range(length)
    plt.plot(steps, dqn_loss, 'r-', alpha=0.7, label='DQN')
    plt.plot(steps, trans_loss, 'b-', alpha=0.7, label='Transformer')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Comparison')
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'enhanced_model_comparison.png'), dpi=300)

# Create detailed bar chart comparison for key metrics
plt.figure(figsize=(15, 10))

metrics_to_compare = [
    {'name': 'Reward', 'dqn': 'rewards', 'transformer': 'rewards', 'better': 'higher'},
    {'name': 'Completion Rate', 'dqn': dqn_completion_key, 'transformer': transformer_completion_key, 'better': 'higher'},
    {'name': 'Latency', 'dqn': dqn_latency_key, 'transformer': transformer_latency_key, 'better': 'lower'},
    {'name': 'Energy', 'dqn': dqn_energy_key, 'transformer': transformer_energy_key, 'better': 'lower'}
]

bar_width = 0.35
index = np.arange(len(metrics_to_compare))
colors_dqn = ['#ff9999', '#ff9999', '#ff9999', '#ff9999']  # Red-ish
colors_transformer = ['#9999ff', '#9999ff', '#9999ff', '#9999ff']  # Blue-ish

dqn_values = []
transformer_values = []
improvement_texts = []

for i, metric in enumerate(metrics_to_compare):
    if metric['dqn'] in dqn_metrics and metric['transformer'] in transformer_metrics:
        dqn_data = dqn_metrics[metric['dqn']]
        transformer_data = transformer_metrics[metric['transformer']]
        
        dqn_avg = np.mean(dqn_data[-100:] if len(dqn_data) >= 100 else dqn_data)
        transformer_avg = np.mean(transformer_data[-100:] if len(transformer_data) >= 100 else transformer_data)
        
        # Normalize values for better comparison
        if metric['name'] == 'Reward':
            dqn_values.append(dqn_avg / 100)
            transformer_values.append(transformer_avg / 100)
        else:
            dqn_values.append(dqn_avg)
            transformer_values.append(transformer_avg)
        
        # Calculate improvement
        diff = transformer_avg - dqn_avg
        if metric['better'] == 'lower':
            improvement = (dqn_avg - transformer_avg) / dqn_avg * 100 if dqn_avg != 0 else float('inf')
            sign = '+' if improvement > 0 else ''
        else:  # higher is better
            improvement = (transformer_avg - dqn_avg) / dqn_avg * 100 if dqn_avg != 0 else float('inf')
            sign = '+' if improvement > 0 else ''
        
        improvement_texts.append(f"{sign}{improvement:.2f}%")

plt.bar(index, dqn_values, bar_width, label='DQN', color=colors_dqn)
plt.bar(index + bar_width, transformer_values, bar_width, label='Transformer', color=colors_transformer)

# Add values and improvement texts
for i in range(len(dqn_values)):
    plt.text(i, dqn_values[i] + 0.02, f"{dqn_values[i]:.4f}", ha='center', va='bottom')
    plt.text(i + bar_width, transformer_values[i] + 0.02, f"{transformer_values[i]:.4f}", ha='center', va='bottom')
    plt.text(i + bar_width/2, max(dqn_values[i], transformer_values[i]) + 0.1, improvement_texts[i], ha='center', va='bottom', fontweight='bold')

plt.ylabel('Value (normalized for Reward)')
plt.title('Performance Comparison: DQN vs Transformer')
plt.xticks(index + bar_width / 2, [m['name'] for m in metrics_to_compare])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'performance_bar_chart.png'), dpi=300)

# Calculate convergence speed comparison
def calculate_convergence(rewards, threshold_pct=0.95):
    """Calculate how many episodes it takes to reach threshold% of max reward"""
    if not rewards:
        return None
    
    max_reward = np.max(rewards)
    threshold = max_reward * threshold_pct
    
    for i, r in enumerate(rewards):
        if r >= threshold:
            return i
    
    return len(rewards)  # Never converged

dqn_convergence = None
transformer_convergence = None

if 'rewards' in dqn_metrics:
    dqn_convergence = calculate_convergence(dqn_metrics['rewards'])
    
if 'rewards' in transformer_metrics:
    transformer_convergence = calculate_convergence(transformer_metrics['rewards'])

# Create advanced metrics table
print("\nAdvanced Performance Metrics:")
print("=" * 60)

advanced_metrics = {
    'Avg Reward (last 100 episodes)': {
        'dqn': np.mean(dqn_metrics['rewards'][-100:]) if 'rewards' in dqn_metrics else 'N/A',
        'transformer': np.mean(transformer_metrics['rewards'][-100:]) if 'rewards' in transformer_metrics else 'N/A',
    },
    'Best Reward': {
        'dqn': np.max(dqn_metrics['rewards']) if 'rewards' in dqn_metrics else 'N/A',
        'transformer': np.max(transformer_metrics['rewards']) if 'rewards' in transformer_metrics else 'N/A',
    },
    'Reward Variance': {
        'dqn': np.var(dqn_metrics['rewards'][-100:]) if 'rewards' in dqn_metrics else 'N/A',
        'transformer': np.var(transformer_metrics['rewards'][-100:]) if 'rewards' in transformer_metrics else 'N/A',
    },
    'Convergence Episode (95% of max)': {
        'dqn': dqn_convergence,
        'transformer': transformer_convergence,
    },
    'Completion Rate': {
        'dqn': np.mean(dqn_metrics[dqn_completion_key][-100:]) if dqn_completion_key else 'N/A',
        'transformer': np.mean(transformer_metrics[transformer_completion_key][-100:]) if transformer_completion_key else 'N/A',
    },
    'Energy Efficiency': {
        'dqn': np.mean(dqn_metrics[dqn_energy_key][-100:]) if dqn_energy_key else 'N/A',
        'transformer': np.mean(transformer_metrics[transformer_energy_key][-100:]) if transformer_energy_key else 'N/A',
    },
    'Avg Latency': {
        'dqn': np.mean(dqn_metrics[dqn_latency_key][-100:]) if dqn_latency_key else 'N/A',
        'transformer': np.mean(transformer_metrics[transformer_latency_key][-100:]) if transformer_latency_key else 'N/A',
    },
    'Energy-Task Ratio': {
        'dqn': (np.mean(dqn_metrics[dqn_energy_key][-100:]) / np.mean(dqn_metrics[dqn_completion_key][-100:])) 
               if (dqn_energy_key and dqn_completion_key) else 'N/A',
        'transformer': (np.mean(transformer_metrics[transformer_energy_key][-100:]) / np.mean(transformer_metrics[transformer_completion_key][-100:]))
                       if (transformer_energy_key and transformer_completion_key) else 'N/A',
    }
}

# Print and save advanced metrics
print(f"{'Metric':<30} {'DQN':<15} {'Transformer':<15} {'Difference':<15} {'Improvement':<15}")
print("-" * 85)

advanced_data = {}
for name, values in advanced_metrics.items():
    if values['dqn'] != 'N/A' and values['transformer'] != 'N/A':
        diff = float(values['transformer']) - float(values['dqn'])
        
        # Determine if higher or lower is better for this metric
        if 'Energy' in name or 'Latency' in name or 'Convergence' in name:
            better = 'lower'
            improvement = (float(values['dqn']) - float(values['transformer'])) / float(values['dqn']) * 100 if float(values['dqn']) != 0 else float('inf')
        else:
            better = 'higher'
            improvement = (float(values['transformer']) - float(values['dqn'])) / float(values['dqn']) * 100 if float(values['dqn']) != 0 else float('inf')
        
        transformer_is_better = (better == 'lower' and diff < 0) or (better == 'higher' and diff > 0)
        
        # Print
        print(f"{name:<30} {float(values['dqn']):<15.4f} {float(values['transformer']):<15.4f} {diff:<15.4f} {improvement:<15.2f}%")
        
        # Store
        advanced_data[name] = {
            'DQN': float(values['dqn']),
            'Transformer': float(values['transformer']),
            'Difference': float(diff),
            'Improvement_percent': float(improvement),
            'Transformer_is_better': bool(transformer_is_better),
            'Better_direction': better
        }
    else:
        print(f"{name:<30} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

# Save advanced metrics as JSON
with open(os.path.join(output_dir, 'advanced_metrics.json'), 'w') as f:
    json.dump(advanced_data, f, indent=4)

# Create specialized comparison: Energy vs Task Completion Rate
plt.figure(figsize=(10, 8))
if dqn_energy_key and transformer_energy_key and dqn_completion_key and transformer_completion_key:
    # Get the last 100 episodes data
    dqn_energy_data = dqn_metrics[dqn_energy_key][-100:]
    dqn_completion_data = dqn_metrics[dqn_completion_key][-100:]
    trans_energy_data = transformer_metrics[transformer_energy_key][-100:]
    trans_completion_data = transformer_metrics[transformer_completion_key][-100:]
    
    # Calculate averages for each 10-episode window to smooth
    window = 10
    dqn_energy_avg = [np.mean(dqn_energy_data[i:i+window]) for i in range(0, len(dqn_energy_data)-window+1, window)]
    dqn_completion_avg = [np.mean(dqn_completion_data[i:i+window]) for i in range(0, len(dqn_completion_data)-window+1, window)]
    trans_energy_avg = [np.mean(trans_energy_data[i:i+window]) for i in range(0, len(trans_energy_data)-window+1, window)]
    trans_completion_avg = [np.mean(trans_completion_data[i:i+window]) for i in range(0, len(trans_completion_data)-window+1, window)]
    
    # Create scatter plot
    plt.scatter(dqn_energy_avg, dqn_completion_avg, color='red', alpha=0.7, s=100, label='DQN')
    plt.scatter(trans_energy_avg, trans_completion_avg, color='blue', alpha=0.7, s=100, label='Transformer')
    
    # Add trend lines
    dqn_z = np.polyfit(dqn_energy_avg, dqn_completion_avg, 1)
    dqn_p = np.poly1d(dqn_z)
    plt.plot(dqn_energy_avg, dqn_p(dqn_energy_avg), "r--", alpha=0.5)
    
    trans_z = np.polyfit(trans_energy_avg, trans_completion_avg, 1)
    trans_p = np.poly1d(trans_z)
    plt.plot(trans_energy_avg, trans_p(trans_energy_avg), "b--", alpha=0.5)
    
    plt.xlabel('Energy Consumption (J)')
    plt.ylabel('Task Completion Rate')
    plt.title('Energy Efficiency vs Task Completion Trade-off')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_vs_completion.png'), dpi=300)

print("\nEnhanced comparison visualizations and metrics saved to:", output_dir)