#!/usr/bin/env python3
"""
Stress test for VEC models with extremely high task load
With additional debugging and error handling
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import torch
import random
import traceback
from collections import defaultdict
import time

from vec_environment import VECEnvironment
from ff_dqn_agent import DQNAgent
from te_dqn_agent import VECTransformerAgent, flatten_observation as te_flatten_observation, get_state_size

def create_stress_test_env(task_intensity='medium'):
    """
    Create an environment with high task load
    
    Args:
        task_intensity: 'low', 'medium', or 'high' to control task generation rate
    """
    # Define task intensity levels
    intensity_settings = {
        'low': {
            'min_tasks': 10,
            'max_tasks': 20,
        },
        'medium': {
            'min_tasks': 15,
            'max_tasks': 30,
        },
        'high': {
            'min_tasks': 25,
            'max_tasks': 40,
        }
    }
    
    # Use medium intensity by default
    settings = intensity_settings.get(task_intensity, intensity_settings['medium'])
    
    print(f"Creating environment with {task_intensity} task intensity:")
    print(f"  - Min tasks per step: {settings['min_tasks']}")
    print(f"  - Max tasks per step: {settings['max_tasks']}")
    
    env = VECEnvironment(
        sumo_config='astana.sumocfg',
        simulation_duration=300,
        time_step=1,
        queue_process_interval=5,
        max_queue_length=100,  # Increased from default 50
        history_length=10,
        energy_csv_path='merged_dag1.csv',
        energy_weight=0.5,
        # Task generation settings based on intensity level
        min_tasks_per_step=settings['min_tasks'],
        max_tasks_per_step=settings['max_tasks'],
        task_generation_probability=1.0,  # Always generate tasks
        seed=999
    )
    
    # Try to modify base stations to have fewer active nodes
    try:
        for bs_id, bs_instance in env.base_station_instances.items():
            original_min = bs_instance.min_active_nodes
            bs_instance.min_active_nodes = 6  # Reduced from default but not too low
            
            print(f"Modified base station {bs_id}: min_active_nodes {original_min} -> {bs_instance.min_active_nodes}")
            
            # Reset active nodes to match new minimum
            active_count = 0
            for i, node in enumerate(bs_instance.nodes):
                node.active = (i < bs_instance.min_active_nodes)
                if node.active:
                    active_count += 1
            
            print(f"Base station {bs_id} has {active_count} active nodes")
    except Exception as e:
        print(f"Warning: Could not modify min_active_nodes: {str(e)}")
        traceback.print_exc()
    
    return env

def run_stress_test(ff_model_path, te_model_path, output_dir='stress_test_results', 
                   episodes=3, max_steps=600, task_intensity='medium'):
    """Run stress test evaluation on both models"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = os.path.join(output_dir, f"stress_test_{timestamp}")
    os.makedirs(test_dir, exist_ok=True)
    
    # Save run configuration
    config = {
        'ff_model_path': ff_model_path,
        'te_model_path': te_model_path,
        'episodes': episodes,
        'max_steps': max_steps,
        'task_intensity': task_intensity,
        'timestamp': timestamp
    }
    
    with open(os.path.join(test_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Starting stress test with {task_intensity} task load...")
    print(f"Results will be saved to: {test_dir}")
    
    # Create environment
    try:
        env = create_stress_test_env(task_intensity)
        
        # Get state and action sizes
        state_size = get_state_size(env)
        action_size = env.action_space.n
        
        print(f"Environment created - State size: {state_size}, Action size: {action_size}")
        
        # Load FF-DQN model
        ff_dqn = DQNAgent(state_size, action_size)
        if os.path.exists(ff_model_path):
            ff_dqn.load_model(ff_model_path)
            print(f"Loaded FF-DQN model from {ff_model_path}")
        else:
            raise FileNotFoundError(f"FF-DQN model not found at {ff_model_path}")
        
        # Load TE-DQN model
        te_dqn = VECTransformerAgent(state_size, action_size)
        if os.path.exists(te_model_path):
            te_dqn.load_model(te_model_path)
            print(f"Loaded TE-DQN model from {te_model_path}")
        else:
            raise FileNotFoundError(f"TE-DQN model not found at {te_model_path}")
        
        # Evaluate models
        print("\n=== Evaluating FF-DQN ===")
        ff_results = evaluate_model(ff_dqn, "FF-DQN", env, episodes, max_steps)
        
        print("\nResetting environment for TE-DQN evaluation...")
        # Reset environment for fair comparison
        env.close()
        env = create_stress_test_env(task_intensity)
        
        print("\n=== Evaluating TE-DQN ===")
        te_results = evaluate_model(te_dqn, "TE-DQN", env, episodes, max_steps)
        
        # Close environment
        env.close()
        
        # Save results
        with open(os.path.join(test_dir, "ff_dqn_results.json"), 'w') as f:
            json.dump(ff_results, f, indent=4)
        
        with open(os.path.join(test_dir, "te_dqn_results.json"), 'w') as f:
            json.dump(te_results, f, indent=4)
        
        # Generate comparative visualizations
        generate_comparisons(ff_results, te_results, test_dir)
        
        print(f"\nStress test completed! Results saved to: {test_dir}")
        
        return test_dir, ff_results, te_results
    
    except Exception as e:
        print(f"Error during stress test: {str(e)}")
        traceback.print_exc()
        
        # Create error report
        with open(os.path.join(test_dir, "error_report.txt"), 'w') as f:
            f.write(f"Error occurred during stress test: {str(e)}\n\n")
            f.write(traceback.format_exc())
        
        print(f"Error report saved to: {os.path.join(test_dir, 'error_report.txt')}")
        
        return test_dir, None, None

def evaluate_model(model, model_name, env, num_episodes=3, max_steps=300):
    """Evaluate a model on the stress test environment"""
    results = {
        "model": model_name,
        "rewards": [],
        "completion_rates": [],
        "rejection_rates": [],
        "drop_rates": [],
        "avg_latencies": [],
        "energy_consumptions": [],
        "idle_energies": [],
        "wake_decisions": [],
        "active_nodes": [],
        "queue_lengths": [],
        "termination_steps": [],  # Track when episodes terminate
        "episode_data": []
    }
    
    for episode in range(num_episodes):
        print(f"Running {model_name} stress test episode {episode+1}/{num_episodes}")
        
        # Reset environment
        obs = env.reset()
        
        # For TE-DQN, we need to clear the state history
        if model_name == "TE-DQN":
            model.state_history.clear()
        
        # Prepare for episode
        episode_reward = 0
        episode_steps = 0
        episode_wake_decisions = 0
        step_data = []
        
        # Track queue length over time
        queue_lengths = []
        active_node_counts = []
        
        # Debug info for each step
        print(f"Starting episode {episode+1}, max steps: {max_steps}")
        step_start_time = time.time()
        
        try:
            while episode_steps < max_steps:
                # Print period debug info
                if episode_steps % 10 == 0:
                    elapsed = time.time() - step_start_time
                    print(f"  Step {episode_steps}/{max_steps}, elapsed: {elapsed:.2f}s")
                
                # Flatten observation based on model type
                if model_name == "FF-DQN":
                    state = model.flatten_observation(obs)
                    action = model.select_action(state)
                else:  # TE-DQN
                    state = te_flatten_observation(obs)
                    action = model.select_action(state, evaluate=True)
                
                # Record if this is a wake-up action
                is_wake_action = (action == env.action_space.n - 1)
                if is_wake_action:
                    episode_wake_decisions += 1
                    print(f"  {model_name} made wake decision at step {episode_steps}")
                
                # Take action in environment
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Track queue lengths across all base stations
                total_queued = sum(len(bs.queue) for bs in env.base_station_instances.values())
                queue_lengths.append(total_queued)
                
                # Track active nodes
                active_nodes = 0
                total_nodes = 0
                for bs_id, bs_instance in env.base_station_instances.items():
                    active_nodes += sum(1 for node in bs_instance.nodes if node.active)
                    total_nodes += len(bs_instance.nodes)
                active_node_ratio = active_nodes / total_nodes if total_nodes > 0 else 0
                active_node_counts.append(active_node_ratio)
                
                # Record detailed step data
                step_data.append({
                    "step": episode_steps,
                    "action": int(action),
                    "reward": float(reward),
                    "is_wake_action": bool(is_wake_action),
                    "task_completion_rate": float(info.get("task_completion_rate", 0)),
                    "task_rejection_rate": float(info.get("task_rejection_rate", 0)),
                    "task_drop_rate": float(info.get("task_drop_rate", 0)),
                    "avg_latency": float(info.get("avg_latency", 0)),
                    "energy_consumption": float(info.get("energy_consumption", 0)),
                    "idle_energy": float(info.get("idle_energy", 0)),
                    "queue_length": int(total_queued),
                    "active_node_ratio": float(active_node_ratio),
                    "simulation_step": info.get("simulation_step", episode_steps)
                })
                
                obs = next_obs
                episode_steps += 1
                
                if done:
                    print(f"  Episode terminated at step {episode_steps} (done signal received)")
                    break
                
                # Check environment state
                if episode_steps % 50 == 0:
                    active_count = 0
                    for bs_id, bs_instance in env.base_station_instances.items():
                        bs_active = sum(1 for node in bs_instance.nodes if node.active)
                        active_count += bs_active
                        print(f"  Base station {bs_id}: {bs_active} active nodes, queue: {len(bs_instance.queue)}")
                    print(f"  Total active nodes: {active_count}")
            
            # Record termination step
            results["termination_steps"].append(episode_steps)
            print(f"Episode {episode+1} completed after {episode_steps} steps")
        
        except Exception as e:
            print(f"Error during episode {episode+1}: {str(e)}")
            traceback.print_exc()
            # Still try to collect whatever data we have
            results["termination_steps"].append(episode_steps)
            print(f"Episode {episode+1} terminated with error after {episode_steps} steps")
        
        # Collect episode metrics
        results["rewards"].append(episode_reward)
        results["completion_rates"].append(info.get("task_completion_rate", 0) if 'info' in locals() else 0)
        results["rejection_rates"].append(info.get("task_rejection_rate", 0) if 'info' in locals() else 0)
        results["drop_rates"].append(info.get("task_drop_rate", 0) if 'info' in locals() else 0)
        results["avg_latencies"].append(info.get("avg_latency", 0) if 'info' in locals() else 0)
        results["energy_consumptions"].append(info.get("avg_energy_consumption", 0) if 'info' in locals() else 0)
        results["idle_energies"].append(info.get("idle_energy", 0) if 'info' in locals() else 0)
        results["wake_decisions"].append(episode_wake_decisions)
        results["active_nodes"].append(np.mean(active_node_counts) if active_node_counts else 0)
        results["queue_lengths"].append(np.mean(queue_lengths) if queue_lengths else 0)
        
        # Store detailed episode data
        results["episode_data"].append({
            "episode": episode,
            "steps": episode_steps,
            "reward": episode_reward,
            "completion_rate": info.get("task_completion_rate", 0) if 'info' in locals() else 0,
            "step_data": step_data
        })
    
    # Calculate average metrics
    avg_results = {
        "avg_reward": float(np.mean(results["rewards"])),
        "std_reward": float(np.std(results["rewards"])),
        "avg_completion_rate": float(np.mean(results["completion_rates"])),
        "avg_rejection_rate": float(np.mean(results["rejection_rates"])),
        "avg_drop_rate": float(np.mean(results["drop_rates"])),
        "avg_latency": float(np.mean(results["avg_latencies"])),
        "avg_energy_consumption": float(np.mean(results["energy_consumptions"])),
        "avg_idle_energy": float(np.mean(results["idle_energies"])),
        "avg_wake_decisions": float(np.mean(results["wake_decisions"])),
        "avg_active_nodes": float(np.mean(results["active_nodes"])),
        "avg_queue_length": float(np.mean(results["queue_lengths"])),
        "avg_steps_per_episode": float(np.mean(results["termination_steps"]))
    }
    
    # Add averages to results
    results["avg_metrics"] = avg_results
    
    return results
def generate_comparisons(ff_results, te_results, output_dir):
    """Generate comprehensive comparison visualizations for stress test"""
    if ff_results is None or te_results is None:
        print("Cannot generate comparisons: missing results")
        return
    
    # Set up the style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Wake Decision Comparison
    plt.figure(figsize=(10, 6))
    model_names = ["FF-DQN", "TE-DQN"]
    wake_decisions = [
        np.mean(ff_results["wake_decisions"]), 
        np.mean(te_results["wake_decisions"])
    ]
    plt.bar(model_names, wake_decisions, color=['blue', 'orange'])
    plt.ylabel("Average Wake Decisions per Episode")
    plt.title("Wake Decisions Under High Load")
    for i, v in enumerate(wake_decisions):
        plt.text(i, v + 0.5, f"{v:.2f}", ha='center')
    plt.savefig(os.path.join(output_dir, "wake_decisions_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance Metrics Comparison
    metrics = {
        "Avg Reward": [ff_results["avg_metrics"]["avg_reward"], te_results["avg_metrics"]["avg_reward"]],
        "Completion Rate": [ff_results["avg_metrics"]["avg_completion_rate"], te_results["avg_metrics"]["avg_completion_rate"]],
        "Rejection Rate": [ff_results["avg_metrics"]["avg_rejection_rate"], te_results["avg_metrics"]["avg_rejection_rate"]],
        "Latency": [ff_results["avg_metrics"]["avg_latency"], te_results["avg_metrics"]["avg_latency"]],
        "Energy": [ff_results["avg_metrics"]["avg_energy_consumption"], te_results["avg_metrics"]["avg_energy_consumption"]],
        "Queue Length": [ff_results["avg_metrics"]["avg_queue_length"], te_results["avg_metrics"]["avg_queue_length"]]
    }
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width/2, [m[0] for m in metrics.values()], width, label="FF-DQN")
    bars2 = ax.bar(x + width/2, [m[1] for m in metrics.values()], width, label="TE-DQN")
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.legend()
    ax.set_title("Performance Metrics Under High Load")
    
    # Add percentage difference labels
    for i, (metric, values) in enumerate(metrics.items()):
        if values[0] > 0:  # Avoid division by zero
            pct_diff = ((values[1] - values[0]) / values[0]) * 100
            
            # For metrics where lower is better, flip the sign
            if metric in ["Rejection Rate", "Latency", "Energy", "Queue Length"]:
                pct_diff = -pct_diff
                
            # Format with + sign for positive values
            pct_text = f"{pct_diff:+.1f}%" if pct_diff != 0 else "0%"
            
            # Change color based on who's better
            color = 'green' if pct_diff > 0 else 'red' if pct_diff < 0 else 'black'
            
            # Place the text above the higher bar
            y_pos = max(values[0], values[1]) + 0.1
            ax.text(i, y_pos, pct_text, ha='center', va='bottom', color=color, fontweight='bold')
    
    plt.savefig(os.path.join(output_dir, "performance_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Queue Length Over Time (for first episode if available)
    plt.figure(figsize=(12, 6))
    
    if (ff_results["episode_data"] and len(ff_results["episode_data"]) > 0 and 
        te_results["episode_data"] and len(te_results["episode_data"]) > 0):
        
        # Extract queue lengths over time
        ff_steps = [step_data["queue_length"] for step_data in ff_results["episode_data"][0]["step_data"]]
        ff_steps_indices = range(len(ff_steps))
        plt.plot(ff_steps_indices, ff_steps, 'b-', label='FF-DQN Queue Length')
        
        te_steps = [step_data["queue_length"] for step_data in te_results["episode_data"][0]["step_data"]]
        te_steps_indices = range(len(te_steps))
        plt.plot(te_steps_indices, te_steps, 'r-', label='TE-DQN Queue Length')
        
        plt.xlabel("Simulation Steps")
        plt.ylabel("Queue Length")
        plt.title("Queue Length Over Time (First Episode)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "queue_length_over_time.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Active Nodes Over Time (for first episode if available)
    plt.figure(figsize=(12, 6))
    
    if (ff_results["episode_data"] and len(ff_results["episode_data"]) > 0 and 
        te_results["episode_data"] and len(te_results["episode_data"]) > 0):
        
        # Extract active nodes over time
        ff_active = [step_data["active_node_ratio"] for step_data in ff_results["episode_data"][0]["step_data"]]
        ff_active_indices = range(len(ff_active))
        plt.plot(ff_active_indices, ff_active, 'b-', label='FF-DQN Active Nodes Ratio')
        
        te_active = [step_data["active_node_ratio"] for step_data in te_results["episode_data"][0]["step_data"]]
        te_active_indices = range(len(te_active))
        plt.plot(te_active_indices, te_active, 'r-', label='TE-DQN Active Nodes Ratio')
        
        plt.xlabel("Simulation Steps")
        plt.ylabel("Active Nodes Ratio")
        plt.title("Active Nodes Over Time (First Episode)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "active_nodes_over_time.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Wake Decision Timing (if data is available)
    plt.figure(figsize=(12, 6))
    
    if (ff_results["episode_data"] and len(ff_results["episode_data"]) > 0 and 
        te_results["episode_data"] and len(te_results["episode_data"]) > 0):
        
        # Extract wake decisions over time
        ff_wake_steps = []
        for step_idx, step_data in enumerate(ff_results["episode_data"][0]["step_data"]):
            if step_data.get("is_wake_action", False):
                ff_wake_steps.append(step_idx)
        
        te_wake_steps = []
        for step_idx, step_data in enumerate(te_results["episode_data"][0]["step_data"]):
            if step_data.get("is_wake_action", False):
                te_wake_steps.append(step_idx)
        
        # Plot vertical lines at wake decision points
        plt.figure(figsize=(12, 6))
        
        # Plot queue lengths as background context
        if len(ff_results["episode_data"][0]["step_data"]) > 0:
            queue_lengths = [step_data["queue_length"] for step_data in ff_results["episode_data"][0]["step_data"]]
            plt.plot(range(len(queue_lengths)), queue_lengths, 'k-', alpha=0.2, label='Queue Length')
        
        for step in ff_wake_steps:
            plt.axvline(x=step, color='blue', linestyle='--', alpha=0.7)
        
        for step in te_wake_steps:
            plt.axvline(x=step, color='red', linestyle='--', alpha=0.7)
        
        plt.xlabel("Simulation Steps")
        plt.ylabel("Queue Length")
        plt.title("Wake Decision Timing with Queue Context (First Episode)")
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', alpha=0.2, label='Queue Length'),
            Line2D([0], [0], color='blue', linestyle='--', label='FF-DQN Wake'),
            Line2D([0], [0], color='red', linestyle='--', label='TE-DQN Wake')
        ]
        plt.legend(handles=legend_elements)
        
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "wake_decision_timing.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. NEW: Reward Distribution
    plt.figure(figsize=(10, 6))
    
    plt.boxplot([ff_results["rewards"], te_results["rewards"]], 
                labels=["FF-DQN", "TE-DQN"],
                patch_artist=True,
                boxprops=dict(facecolor="lightblue"),
                medianprops=dict(color="red"))
    
    plt.ylabel("Episode Reward")
    plt.title("Reward Distribution Comparison")
    plt.grid(True, axis='y')
    
    # Add individual points
    for i, rewards in enumerate([ff_results["rewards"], te_results["rewards"]]):
        x = np.random.normal(i+1, 0.04, size=len(rewards))
        plt.scatter(x, rewards, alpha=0.6, s=40)
    
    plt.savefig(os.path.join(output_dir, "reward_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. NEW: Energy vs Latency Tradeoff Scatter Plot
    plt.figure(figsize=(10, 8))
    
    # Get step data for energy and latency
    if (ff_results["episode_data"] and te_results["episode_data"]):
        # FF-DQN data
        ff_latencies = []
        ff_energies = []
        for episode_data in ff_results["episode_data"]:
            for step_data in episode_data["step_data"]:
                if step_data["avg_latency"] > 0:  # Only include valid data points
                    ff_latencies.append(step_data["avg_latency"])
                    ff_energies.append(step_data["energy_consumption"])
        
        # TE-DQN data
        te_latencies = []
        te_energies = []
        for episode_data in te_results["episode_data"]:
            for step_data in episode_data["step_data"]:
                if step_data["avg_latency"] > 0:  # Only include valid data points
                    te_latencies.append(step_data["avg_latency"])
                    te_energies.append(step_data["energy_consumption"])
        
        # Plot scatter
        plt.scatter(ff_latencies, ff_energies, alpha=0.5, label="FF-DQN", color="blue", s=30)
        plt.scatter(te_latencies, te_energies, alpha=0.5, label="TE-DQN", color="orange", s=30)
        
        # Add centroids
        if ff_latencies and ff_energies:
            plt.scatter(np.mean(ff_latencies), np.mean(ff_energies), color="darkblue", 
                       s=100, marker='*', label="FF-DQN Centroid")
        if te_latencies and te_energies:
            plt.scatter(np.mean(te_latencies), np.mean(te_energies), color="darkred", 
                       s=100, marker='*', label="TE-DQN Centroid")
        
        plt.xlabel("Latency (s)")
        plt.ylabel("Energy Consumption (J)")
        plt.title("Energy-Latency Tradeoff")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "energy_latency_tradeoff.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. NEW: Action Distribution (Node Selection)
    plt.figure(figsize=(12, 6))
    
    if (ff_results["episode_data"] and te_results["episode_data"]):
        # Extract action frequencies for both models
        ff_actions = []
        te_actions = []
        
        for episode_data in ff_results["episode_data"]:
            for step_data in episode_data["step_data"]:
                ff_actions.append(step_data["action"])
        
        for episode_data in te_results["episode_data"]:
            for step_data in episode_data["step_data"]:
                te_actions.append(step_data["action"])
        
        # Count frequencies
        action_space = max(max(ff_actions) if ff_actions else 0, 
                          max(te_actions) if te_actions else 0) + 1
        
        ff_counts = np.zeros(action_space)
        te_counts = np.zeros(action_space)
        
        for action in ff_actions:
            ff_counts[action] += 1
        
        for action in te_actions:
            te_counts[action] += 1
        
        # Normalize to percentages
        if len(ff_actions) > 0:
            ff_pct = (ff_counts / len(ff_actions)) * 100
        else:
            ff_pct = ff_counts
            
        if len(te_actions) > 0:
            te_pct = (te_counts / len(te_actions)) * 100
        else:
            te_pct = te_counts
        
        # Plot
        x = np.arange(action_space)
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.bar(x - width/2, ff_pct, width, label="FF-DQN")
        ax.bar(x + width/2, te_pct, width, label="TE-DQN")
        
        # Mark the wake-up action
        wake_idx = action_space - 1
        ax.axvline(x=wake_idx, color='red', linestyle='--', alpha=0.3)
        ax.text(wake_idx, max(max(ff_pct), max(te_pct)) * 0.9, "Wake Action", 
               rotation=90, ha='center', va='top', color='red')
        
        ax.set_xlabel("Action (Node ID)")
        ax.set_ylabel("Selection Frequency (%)")
        ax.set_title("Action Distribution")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(action_space)])
        ax.legend()
        
        plt.savefig(os.path.join(output_dir, "action_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. NEW: Reward Over Time (First Episode)
    plt.figure(figsize=(12, 6))
    
    if (ff_results["episode_data"] and len(ff_results["episode_data"]) > 0 and 
        te_results["episode_data"] and len(te_results["episode_data"]) > 0):
        
        # Extract reward over time from first episode
        ff_rewards = [step_data["reward"] for step_data in ff_results["episode_data"][0]["step_data"]]
        ff_cumulative = np.cumsum(ff_rewards)
        
        te_rewards = [step_data["reward"] for step_data in te_results["episode_data"][0]["step_data"]]
        te_cumulative = np.cumsum(te_rewards)
        
        plt.figure(figsize=(12, 6))
        
        # Plot instantaneous rewards
        plt.subplot(1, 2, 1)
        plt.plot(range(len(ff_rewards)), ff_rewards, 'b-', alpha=0.7, label='FF-DQN')
        plt.plot(range(len(te_rewards)), te_rewards, 'r-', alpha=0.7, label='TE-DQN')
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Instantaneous Rewards (First Episode)")
        plt.legend()
        plt.grid(True)
        
        # Plot cumulative rewards
        plt.subplot(1, 2, 2)
        plt.plot(range(len(ff_cumulative)), ff_cumulative, 'b-', label='FF-DQN')
        plt.plot(range(len(te_cumulative)), te_cumulative, 'r-', label='TE-DQN')
        plt.xlabel("Step")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Rewards (First Episode)")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "reward_over_time.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 10. NEW: Radar Chart of Key Metrics
    plt.figure(figsize=(10, 10))
    
    # Define the metrics for the radar chart
    radar_metrics = [
        "Completion Rate",
        "Energy Efficiency",  # Inverse of energy consumption
        "Latency Efficiency", # Inverse of latency
        "Node Efficiency",    # Ratio of active nodes to total
        "Wake Decision Efficiency" # Fewer is better
    ]
    
    # Calculate values (normalized between 0 and 1)
    # For metrics where higher is better:
    ff_completion = ff_results["avg_metrics"]["avg_completion_rate"]
    te_completion = te_results["avg_metrics"]["avg_completion_rate"]
    
    # For metrics where lower is better, invert them so higher is better in the chart
    # Energy efficiency (inverse of energy consumption)
    max_energy = max(ff_results["avg_metrics"]["avg_energy_consumption"], 
                     te_results["avg_metrics"]["avg_energy_consumption"])
    ff_energy_eff = 1 - (ff_results["avg_metrics"]["avg_energy_consumption"] / max_energy if max_energy > 0 else 0)
    te_energy_eff = 1 - (te_results["avg_metrics"]["avg_energy_consumption"] / max_energy if max_energy > 0 else 0)
    
    # Latency efficiency (inverse of latency)
    max_latency = max(ff_results["avg_metrics"]["avg_latency"], 
                      te_results["avg_metrics"]["avg_latency"])
    ff_latency_eff = 1 - (ff_results["avg_metrics"]["avg_latency"] / max_latency if max_latency > 0 else 0)
    te_latency_eff = 1 - (te_results["avg_metrics"]["avg_latency"] / max_latency if max_latency > 0 else 0)
    
    # Node efficiency (ratio of active nodes)
    ff_node_eff = 1 - ff_results["avg_metrics"]["avg_active_nodes"]
    te_node_eff = 1 - te_results["avg_metrics"]["avg_active_nodes"]
    
    # Wake decision efficiency (fewer is better)
    max_wake = max(ff_results["avg_metrics"]["avg_wake_decisions"], 
                   te_results["avg_metrics"]["avg_wake_decisions"])
    ff_wake_eff = 1 - (ff_results["avg_metrics"]["avg_wake_decisions"] / max_wake if max_wake > 0 else 0)
    te_wake_eff = 1 - (te_results["avg_metrics"]["avg_wake_decisions"] / max_wake if max_wake > 0 else 0)
    
    # Combine values
    ff_values = [ff_completion, ff_energy_eff, ff_latency_eff, ff_node_eff, ff_wake_eff]
    te_values = [te_completion, te_energy_eff, te_latency_eff, te_node_eff, te_wake_eff]
    
    # Make sure values are between 0 and 1
    ff_values = [max(0, min(1, v)) for v in ff_values]
    te_values = [max(0, min(1, v)) for v in te_values]
    
    # Number of variables
    N = len(radar_metrics)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add the values for the first point to close the loop
    ff_values += ff_values[:1]
    te_values += te_values[:1]
    
    # Create the plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], radar_metrics, size=12)
    
    # Draw the outline of the data
    ax.plot(angles, ff_values, 'b-', linewidth=2, label='FF-DQN')
    ax.fill(angles, ff_values, 'b', alpha=0.1)
    
    ax.plot(angles, te_values, 'r-', linewidth=2, label='TE-DQN')
    ax.fill(angles, te_values, 'r', alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title("Performance Radar Chart", size=15)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "radar_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary text file
    with open(os.path.join(output_dir, "stress_test_summary.txt"), 'w') as f:
        f.write("Stress Test Summary\n")
        f.write("=================\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Episode Lengths:\n")
        f.write(f"  FF-DQN: {np.mean(ff_results['termination_steps']):.2f} steps per episode\n")
        f.write(f"  TE-DQN: {np.mean(te_results['termination_steps']):.2f} steps per episode\n\n")
        
        f.write("Wake Decisions:\n")
        f.write(f"  FF-DQN: {np.mean(ff_results['wake_decisions']):.2f} per episode\n")
        f.write(f"  TE-DQN: {np.mean(te_results['wake_decisions']):.2f} per episode\n\n")
        
        f.write("Performance Metrics:\n")
        for metric, values in metrics.items():
            f.write(f"  {metric}:\n")
            f.write(f"    FF-DQN: {values[0]:.4f}\n")
            f.write(f"    TE-DQN: {values[1]:.4f}\n")
            diff = values[1] - values[0]
            pct = (diff / values[0]) * 100 if values[0] != 0 else 0
            better = "TE-DQN" if diff > 0 else "FF-DQN" if diff < 0 else "Equal"
            # For metrics where lower is better, flip the comparison
            if metric in ["Rejection Rate", "Latency", "Energy", "Queue Length"]:
                better = "TE-DQN" if diff < 0 else "FF-DQN" if diff > 0 else "Equal"
                pct = -pct
            f.write(f"    Difference: {diff:.4f} ({pct:.2f}%), Better: {better}\n\n")
        
        f.write("\nDetailed Wake Decisions:\n")
        if ff_results["episode_data"]:
            f.write("  FF-DQN Wake Steps (Episode 1):\n")
            wake_steps = []
            for step_idx, step_data in enumerate(ff_results["episode_data"][0]["step_data"]):
                if step_data.get("is_wake_action", False):
                    wake_steps.append(str(step_idx))
            f.write("    " + ", ".join(wake_steps) + "\n\n")
        
        if te_results["episode_data"]:
            f.write("  TE-DQN Wake Steps (Episode 1):\n")
            wake_steps = []
            for step_idx, step_data in enumerate(te_results["episode_data"][0]["step_data"]):
                if step_data.get("is_wake_action", False):
                    wake_steps.append(str(step_idx))
            f.write("    " + ", ".join(wake_steps) + "\n")
        
        # Add analysis section
        f.write("\n\nAnalysis Summary:\n")
        f.write("===============\n\n")
        
        # Analyze task completion
        completion_diff = te_results["avg_metrics"]["avg_completion_rate"] - ff_results["avg_metrics"]["avg_completion_rate"]
        completion_pct = (completion_diff / ff_results["avg_metrics"]["avg_completion_rate"]) * 100 if ff_results["avg_metrics"]["avg_completion_rate"] > 0 else 0
        
        f.write(f"Task Completion: ")
        if abs(completion_pct) < 1:
            f.write(f"Both models have similar task completion rates (difference: {completion_pct:.2f}%).\n")
        else:
            better = "TE-DQN" if completion_pct > 0 else "FF-DQN"
            f.write(f"{better} shows {abs(completion_pct):.2f}% better task completion rate.\n")
        
        # Analyze energy efficiency
        energy_diff = ff_results["avg_metrics"]["avg_energy_consumption"] - te_results["avg_metrics"]["avg_energy_consumption"]
        energy_pct = (energy_diff / ff_results["avg_metrics"]["avg_energy_consumption"]) * 100 if ff_results["avg_metrics"]["avg_energy_consumption"] > 0 else 0
        
        f.write(f"Energy Efficiency: ")
        if energy_diff > 0:
            f.write(f"TE-DQN is more energy efficient, consuming {energy_pct:.2f}% less energy.\n")
        elif energy_diff < 0:
            f.write(f"FF-DQN is more energy efficient, consuming {-energy_pct:.2f}% less energy.\n")
        else:
            f.write(f"Both models have similar energy consumption.\n")
        
        # Analyze latency
        latency_diff = ff_results["avg_metrics"]["avg_latency"] - te_results["avg_metrics"]["avg_latency"]
        latency_pct = (latency_diff / ff_results["avg_metrics"]["avg_latency"]) * 100 if ff_results["avg_metrics"]["avg_latency"] > 0 else 0
        
        f.write(f"Latency: ")
        if latency_diff > 0:
            f.write(f"TE-DQN achieves {latency_pct:.2f}% lower latency.\n")
        elif latency_diff < 0:
            f.write(f"FF-DQN achieves {-latency_pct:.2f}% lower latency.\n")
        else:
            f.write(f"Both models have similar latency performance.\n")
        
        # Overall conclusion
        f.write("\nOverall Performance: ")
        
        advantages_te = []
        advantages_ff = []
        
        for metric, values in metrics.items():
            diff = values[1] - values[0]
            if metric in ["Rejection Rate", "Latency", "Energy", "Queue Length"]:
                # Lower is better
                if diff < -0.01:  # Small threshold to ignore minor differences
                    advantages_te.append(metric)
                elif diff > 0.01:
                    advantages_ff.append(metric)
            else:
                # Higher is better
                if diff > 0.01:
                    advantages_te.append(metric)
                elif diff < -0.01:
                    advantages_ff.append(metric)
        
        if len(advantages_te) > len(advantages_ff):
            f.write(f"TE-DQN outperforms FF-DQN overall, with advantages in {', '.join(advantages_te)}.\n")
        elif len(advantages_ff) > len(advantages_te):
            f.write(f"FF-DQN outperforms TE-DQN overall, with advantages in {', '.join(advantages_ff)}.\n")
        else:
            f.write(f"Both models show comparable overall performance, with different strengths.\n")
            if advantages_te:
                f.write(f"TE-DQN excels in: {', '.join(advantages_te)}.\n")
            if advantages_ff:
                f.write(f"FF-DQN excels in: {', '.join(advantages_ff)}.\n")
# def generate_comparisons(ff_results, te_results, output_dir):
#     """Generate comparison visualizations specifically for stress test"""
#     if ff_results is None or te_results is None:
#         print("Cannot generate comparisons: missing results")
#         return
    
#     # Set up the style
#     plt.style.use('seaborn-v0_8-darkgrid')
    
#     # 1. Wake Decision Comparison
#     plt.figure(figsize=(10, 6))
#     model_names = ["FF-DQN", "TE-DQN"]
#     wake_decisions = [
#         np.mean(ff_results["wake_decisions"]), 
#         np.mean(te_results["wake_decisions"])
#     ]
#     plt.bar(model_names, wake_decisions, color=['blue', 'orange'])
#     plt.ylabel("Average Wake Decisions per Episode")
#     plt.title("Wake Decisions Under High Load")
#     for i, v in enumerate(wake_decisions):
#         plt.text(i, v + 0.5, f"{v:.2f}", ha='center')
#     plt.savefig(os.path.join(output_dir, "wake_decisions_comparison.png"), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # 2. Performance Metrics Comparison
#     metrics = {
#         "Avg Reward": [ff_results["avg_metrics"]["avg_reward"], te_results["avg_metrics"]["avg_reward"]],
#         "Completion Rate": [ff_results["avg_metrics"]["avg_completion_rate"], te_results["avg_metrics"]["avg_completion_rate"]],
#         "Rejection Rate": [ff_results["avg_metrics"]["avg_rejection_rate"], te_results["avg_metrics"]["avg_rejection_rate"]],
#         "Latency": [ff_results["avg_metrics"]["avg_latency"], te_results["avg_metrics"]["avg_latency"]],
#         "Energy": [ff_results["avg_metrics"]["avg_energy_consumption"], te_results["avg_metrics"]["avg_energy_consumption"]],
#         "Queue Length": [ff_results["avg_metrics"]["avg_queue_length"], te_results["avg_metrics"]["avg_queue_length"]]
#     }
    
#     plt.figure(figsize=(14, 8))
#     x = np.arange(len(metrics))
#     width = 0.35
    
#     fig, ax = plt.subplots(figsize=(14, 8))
#     ax.bar(x - width/2, [m[0] for m in metrics.values()], width, label="FF-DQN")
#     ax.bar(x + width/2, [m[1] for m in metrics.values()], width, label="TE-DQN")
    
#     ax.set_xticks(x)
#     ax.set_xticklabels(metrics.keys())
#     ax.legend()
#     ax.set_title("Performance Metrics Under High Load")
    
#     plt.savefig(os.path.join(output_dir, "performance_metrics.png"), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # 3. Queue Length Over Time (for first episode if available)
#     plt.figure(figsize=(12, 6))
    
#     if (ff_results["episode_data"] and len(ff_results["episode_data"]) > 0 and 
#         te_results["episode_data"] and len(te_results["episode_data"]) > 0):
        
#         # Extract queue lengths over time
#         ff_steps = [step_data["queue_length"] for step_data in ff_results["episode_data"][0]["step_data"]]
#         ff_steps_indices = range(len(ff_steps))
#         plt.plot(ff_steps_indices, ff_steps, 'b-', label='FF-DQN Queue Length')
        
#         te_steps = [step_data["queue_length"] for step_data in te_results["episode_data"][0]["step_data"]]
#         te_steps_indices = range(len(te_steps))
#         plt.plot(te_steps_indices, te_steps, 'r-', label='TE-DQN Queue Length')
        
#         plt.xlabel("Simulation Steps")
#         plt.ylabel("Queue Length")
#         plt.title("Queue Length Over Time (First Episode)")
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(os.path.join(output_dir, "queue_length_over_time.png"), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # 4. Active Nodes Over Time (for first episode if available)
#     plt.figure(figsize=(12, 6))
    
#     if (ff_results["episode_data"] and len(ff_results["episode_data"]) > 0 and 
#         te_results["episode_data"] and len(te_results["episode_data"]) > 0):
        
#         # Extract active nodes over time
#         ff_active = [step_data["active_node_ratio"] for step_data in ff_results["episode_data"][0]["step_data"]]
#         ff_active_indices = range(len(ff_active))
#         plt.plot(ff_active_indices, ff_active, 'b-', label='FF-DQN Active Nodes Ratio')
        
#         te_active = [step_data["active_node_ratio"] for step_data in te_results["episode_data"][0]["step_data"]]
#         te_active_indices = range(len(te_active))
#         plt.plot(te_active_indices, te_active, 'r-', label='TE-DQN Active Nodes Ratio')
        
#         plt.xlabel("Simulation Steps")
#         plt.ylabel("Active Nodes Ratio")
#         plt.title("Active Nodes Over Time (First Episode)")
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(os.path.join(output_dir, "active_nodes_over_time.png"), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # 5. Wake Decision Timing (if data is available)
#     plt.figure(figsize=(12, 6))
    
#     if (ff_results["episode_data"] and len(ff_results["episode_data"]) > 0 and 
#         te_results["episode_data"] and len(te_results["episode_data"]) > 0):
        
#         # Extract wake decisions over time
#         ff_wake_steps = []
#         for step_idx, step_data in enumerate(ff_results["episode_data"][0]["step_data"]):
#             if step_data.get("is_wake_action", False):
#                 ff_wake_steps.append(step_idx)
        
#         te_wake_steps = []
#         for step_idx, step_data in enumerate(te_results["episode_data"][0]["step_data"]):
#             if step_data.get("is_wake_action", False):
#                 te_wake_steps.append(step_idx)
        
#         # Plot vertical lines at wake decision points
#         y_min, y_max = 0, 1  # Y-axis range for the lines
        
#         for step in ff_wake_steps:
#             plt.axvline(x=step, color='blue', linestyle='--', alpha=0.7)
        
#         for step in te_wake_steps:
#             plt.axvline(x=step, color='red', linestyle='--', alpha=0.7)
        
#         plt.xlabel("Simulation Steps")
#         plt.title("Wake Decision Timing (First Episode)")
#         plt.yticks([])  # Hide y-axis ticks as they're not meaningful
        
#         # Create custom legend even if no wake decisions
#         from matplotlib.lines import Line2D
#         legend_elements = [
#             Line2D([0], [0], color='blue', linestyle='--', label='FF-DQN Wake'),
#             Line2D([0], [0], color='red', linestyle='--', label='TE-DQN Wake')
#         ]
#         plt.legend(handles=legend_elements)
        
#         plt.grid(True)
#         plt.savefig(os.path.join(output_dir, "wake_decision_timing.png"), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Create summary text file
#     with open(os.path.join(output_dir, "stress_test_summary.txt"), 'w') as f:
#         f.write("Stress Test Summary\n")
#         f.write("=================\n\n")
#         f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
#         f.write("Episode Lengths:\n")
#         f.write(f"  FF-DQN: {np.mean(ff_results['termination_steps']):.2f} steps per episode\n")
#         f.write(f"  TE-DQN: {np.mean(te_results['termination_steps']):.2f} steps per episode\n\n")
        
#         f.write("Wake Decisions:\n")
#         f.write(f"  FF-DQN: {np.mean(ff_results['wake_decisions']):.2f} per episode\n")
#         f.write(f"  TE-DQN: {np.mean(te_results['wake_decisions']):.2f} per episode\n\n")
        
#         f.write("Performance Metrics:\n")
#         for metric, values in metrics.items():
#             f.write(f"  {metric}:\n")
#             f.write(f"    FF-DQN: {values[0]:.4f}\n")
#             f.write(f"    TE-DQN: {values[1]:.4f}\n")
#             diff = values[1] - values[0]
#             pct = (diff / values[0]) * 100 if values[0] != 0 else 0
#             better = "TE-DQN" if diff > 0 else "FF-DQN" if diff < 0 else "Equal"
#             # For metrics where lower is better, flip the comparison
#             if metric in ["Rejection Rate", "Latency", "Energy", "Queue Length"]:
#                 better = "TE-DQN" if diff < 0 else "FF-DQN" if diff > 0 else "Equal"
#                 pct = -pct
#             f.write(f"    Difference: {diff:.4f} ({pct:.2f}%), Better: {better}\n\n")
        
#         f.write("\nDetailed Wake Decisions:\n")
#         if ff_results["episode_data"]:
#             f.write("  FF-DQN Wake Steps (Episode 1):\n")
#             wake_steps = []
#             for step_idx, step_data in enumerate(ff_results["episode_data"][0]["step_data"]):
#                 if step_data.get("is_wake_action", False):
#                     wake_steps.append(str(step_idx))
#             f.write("    " + ", ".join(wake_steps) + "\n\n")
        
#         if te_results["episode_data"]:
#             f.write("  TE-DQN Wake Steps (Episode 1):\n")
#             wake_steps = []
#             for step_idx, step_data in enumerate(te_results["episode_data"][0]["step_data"]):
#                 if step_data.get("is_wake_action", False):
#                     wake_steps.append(str(step_idx))
#             f.write("    " + ", ".join(wake_steps) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run stress test evaluation on VEC models")
    parser.add_argument('--ff_model', type=str, required=True, help='Path to FF-DQN model')
    parser.add_argument('--te_model', type=str, required=True, help='Path to TE-DQN model')
    parser.add_argument('--output', type=str, default='stress_test_results', help='Output directory')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--intensity', type=str, default='medium', 
                       choices=['low', 'medium', 'high'], help='Task load intensity')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    test_dir, ff_results, te_results = run_stress_test(
        args.ff_model, 
        args.te_model, 
        args.output,
        args.episodes,
        task_intensity=args.intensity
    )
    
    if ff_results and te_results:
        print(f"Stress test completed! Results saved to: {test_dir}")
        print(f"See {os.path.join(test_dir, 'stress_test_summary.txt')} for summary")
    else:
        print(f"Stress test encountered errors. Check: {test_dir}")

if __name__ == "__main__":
    main()