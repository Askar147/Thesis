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
    """Generate comparison visualizations specifically for stress test"""
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
    ax.bar(x - width/2, [m[0] for m in metrics.values()], width, label="FF-DQN")
    ax.bar(x + width/2, [m[1] for m in metrics.values()], width, label="TE-DQN")
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.legend()
    ax.set_title("Performance Metrics Under High Load")
    
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
        y_min, y_max = 0, 1  # Y-axis range for the lines
        
        for step in ff_wake_steps:
            plt.axvline(x=step, color='blue', linestyle='--', alpha=0.7)
        
        for step in te_wake_steps:
            plt.axvline(x=step, color='red', linestyle='--', alpha=0.7)
        
        plt.xlabel("Simulation Steps")
        plt.title("Wake Decision Timing (First Episode)")
        plt.yticks([])  # Hide y-axis ticks as they're not meaningful
        
        # Create custom legend even if no wake decisions
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linestyle='--', label='FF-DQN Wake'),
            Line2D([0], [0], color='red', linestyle='--', label='TE-DQN Wake')
        ]
        plt.legend(handles=legend_elements)
        
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "wake_decision_timing.png"), dpi=300, bbox_inches='tight')
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