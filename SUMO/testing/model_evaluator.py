import os
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import torch
import random
from scipy import stats

# Import environment and agent classes
from vec_environment import VECEnvironment
from ff_dqn_agent import DQNAgent
from te_dqn_agent import VECTransformerAgent, flatten_observation as te_flatten_observation, get_state_size

class ModelEvaluator:
    """Class for evaluating and comparing VEC offloading models"""
    
    def __init__(self, output_dir="evaluation_results"):
        """Initialize the evaluator"""
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_dir = os.path.join(output_dir, f"eval_{self.timestamp}")
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # Set up scenario configurations
        self.scenarios = {
            "baseline": {
                "name": "Baseline Scenario",
                "description": "Similar to training conditions",
                "config": {
                    "min_tasks_per_step": 5,
                    "max_tasks_per_step": 10,
                    "task_generation_probability": 0.8,
                    "seed": 100
                }
            },
            "high_traffic": {
                "name": "High Traffic Scenario",
                "description": "Increased task generation to simulate heavy traffic",
                "config": {
                    "min_tasks_per_step": 15,
                    "max_tasks_per_step": 30,
                    "task_generation_probability": 1.0,
                    "seed": 101
                }
            },
            "resource_constrained": {
                "name": "Resource Constrained Scenario",
                "description": "Fewer active nodes to test wake-up decisions",
                "config": {
                    "min_tasks_per_step": 10,
                    "max_tasks_per_step": 20,
                    "task_generation_probability": 0.9,
                    "min_active_nodes": 5,  # Reduced from default 10
                    "seed": 102
                }
            },
            "deadline_sensitive": {
                "name": "Deadline Sensitive Scenario",
                "description": "Tight deadlines to stress scheduling",
                "config": {
                    "min_tasks_per_step": 8,
                    "max_tasks_per_step": 16,
                    "task_generation_probability": 0.9,
                    "deadline_multiplier": 0.6,  # Tighter deadlines
                    "seed": 103
                }
            },
            "energy_sensitive": {
                "name": "Energy Sensitive Scenario",
                "description": "Higher energy weight for evaluation",
                "config": {
                    "min_tasks_per_step": 8,
                    "max_tasks_per_step": 16,
                    "task_generation_probability": 0.9,
                    "energy_weight": 0.8,  # Higher energy weight
                    "seed": 104
                }
            }
        }
        
    def load_models(self, ff_dqn_path, te_dqn_path):
        """Load the trained models"""
        # Create a temporary environment to determine state size
        temp_env = VECEnvironment(
            simulation_duration=10,
            energy_csv_path="merged_dag1.csv"
        )
        state_size = get_state_size(temp_env)
        action_size = temp_env.action_space.n
        temp_env.close()
        
        # Load FF-DQN
        self.ff_dqn = DQNAgent(state_size, action_size)
        if os.path.exists(ff_dqn_path):
            self.ff_dqn.load_model(ff_dqn_path)
            print(f"Loaded FF-DQN model from {ff_dqn_path}")
        else:
            raise FileNotFoundError(f"FF-DQN model not found at {ff_dqn_path}")
            
        # Load TE-DQN
        self.te_dqn = VECTransformerAgent(state_size, action_size)
        if os.path.exists(te_dqn_path):
            self.te_dqn.load_model(te_dqn_path)
            print(f"Loaded TE-DQN model from {te_dqn_path}")
        else:
            raise FileNotFoundError(f"TE-DQN model not found at {te_dqn_path}")
    
    def create_env(self, scenario_config, base_config=None):
        """Create environment with specific configuration"""
        # Start with base configuration
        if base_config is None:
            base_config = {
                'sumo_config': 'astana.sumocfg',
                'simulation_duration': 300,
                'time_step': 1,
                'queue_process_interval': 5,
                'max_queue_length': 50,
                'history_length': 10,
                'energy_csv_path': 'merged_dag1.csv',
                'energy_weight': 0.5,
                'latency_model_params': {
                    'frequency_band': 2.4,
                    'bandwidth': 20,
                    'noise_floor': -95
                }
            }
        
        # Update with scenario specific config
        config = {**base_config, **scenario_config}
        
        # Create environment
        env = VECEnvironment(
            sumo_config=config.get('sumo_config', 'astana.sumocfg'),
            simulation_duration=config.get('simulation_duration', 300),
            time_step=config.get('time_step', 1),
            queue_process_interval=config.get('queue_process_interval', 5),
            max_queue_length=config.get('max_queue_length', 50),
            history_length=config.get('history_length', 10),
            energy_csv_path=config.get('energy_csv_path', None),
            energy_weight=config.get('energy_weight', 0.5),
            latency_model_params=config.get('latency_model_params', None),
            min_tasks_per_step=config.get('min_tasks_per_step', 2),
            max_tasks_per_step=config.get('max_tasks_per_step', 10),
            task_generation_probability=config.get('task_generation_probability', 0.8),
            seed=config.get('seed', 42)
        )
        
        return env
    
    def evaluate_model(self, model, model_name, env, num_episodes=10, max_steps=300):
        """Evaluate a model on a specific environment"""
        results = {
            "model": model_name,
            "rewards": [],
            "completion_rates": [],
            "rejection_rates": [],
            "drop_rates": [],
            "avg_latencies": [],
            "energy_consumptions": [],
            "idle_energies": [],
            "active_nodes_count": [],
            "data_rates": [],
            "wake_decisions": [],
            "episode_data": []
        }
        
        for episode in range(num_episodes):
            print(f"Running {model_name} evaluation episode {episode+1}/{num_episodes}")
            
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
            
            # Run episode
            while episode_steps < max_steps:
                # Flatten observation based on model type
                if model_name == "FF-DQN":
                    state = model.flatten_observation(obs)  # Use the method from the instance
                    action = model.select_action(state)
                else:  # TE-DQN
                    state = te_flatten_observation(obs)
                    action = model.select_action(state, evaluate=True)
                
                # Check if this is a wake-up action
                if action == env.action_space.n - 1:
                    episode_wake_decisions += 1
                
                # Take action in environment
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Record detailed step data
                step_data.append({
                    "step": episode_steps,
                    "action": int(action),
                    "reward": float(reward),
                    "task_completion_rate": float(info.get("task_completion_rate", 0)),
                    "task_rejection_rate": float(info.get("task_rejection_rate", 0)),
                    "task_drop_rate": float(info.get("task_drop_rate", 0)),
                    "avg_latency": float(info.get("avg_latency", 0)),
                    "energy_consumption": float(info.get("energy_consumption", 0)),
                    "idle_energy": float(info.get("idle_energy", 0)),
                    "pending_tasks": int(info.get("pending_tasks", 0)),
                    "total_queued_tasks": int(info.get("total_queued_tasks", 0)),
                    "is_wake_action": bool(action == env.action_space.n - 1)
                })
                
                # Update observation
                obs = next_obs
                episode_steps += 1
                
                if done:
                    break
            
            # Collect episode metrics
            results["rewards"].append(episode_reward)
            results["completion_rates"].append(info.get("task_completion_rate", 0))
            results["rejection_rates"].append(info.get("task_rejection_rate", 0))
            results["drop_rates"].append(info.get("task_drop_rate", 0))
            results["avg_latencies"].append(info.get("avg_latency", 0))
            results["energy_consumptions"].append(info.get("avg_energy_consumption", 0))
            results["idle_energies"].append(info.get("idle_energy", 0))
            results["data_rates"].append(info.get("avg_data_rate", 0))
            results["wake_decisions"].append(episode_wake_decisions)
            
            # Count active nodes across all base stations
            active_nodes = 0
            total_nodes = 0
            for bs_id, bs_instance in env.base_station_instances.items():
                active_nodes += sum(1 for node in bs_instance.nodes if node.active)
                total_nodes += len(bs_instance.nodes)
            results["active_nodes_count"].append(active_nodes / total_nodes if total_nodes > 0 else 0)
            
            # Store detailed episode data
            results["episode_data"].append({
                "episode": episode,
                "steps": episode_steps,
                "reward": episode_reward,
                "completion_rate": info.get("task_completion_rate", 0),
                "step_data": step_data
            })
        
        # Calculate average metrics
        avg_results = {
            "avg_reward": np.mean(results["rewards"]),
            "std_reward": np.std(results["rewards"]),
            "avg_completion_rate": np.mean(results["completion_rates"]),
            "avg_rejection_rate": np.mean(results["rejection_rates"]),
            "avg_drop_rate": np.mean(results["drop_rates"]),
            "avg_latency": np.mean(results["avg_latencies"]),
            "avg_energy_consumption": np.mean(results["energy_consumptions"]),
            "avg_idle_energy": np.mean(results["idle_energies"]),
            "avg_active_nodes_ratio": np.mean(results["active_nodes_count"]),
            "avg_wake_decisions": np.mean(results["wake_decisions"]),
            "min_reward": np.min(results["rewards"]),
            "max_reward": np.max(results["rewards"])
        }
        
        # Add averages to results
        results["avg_metrics"] = avg_results
        
        return results
    
    def evaluate_scenario(self, scenario_name, num_episodes=10, max_steps=300):
        """Evaluate both models on a specific scenario"""
        scenario = self.scenarios[scenario_name]
        print(f"\nEvaluating scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        
        # Create environment
        env_config = scenario["config"]
        env = self.create_env(env_config)
        
        # Evaluate FF-DQN
        ff_results = self.evaluate_model(self.ff_dqn, "FF-DQN", env, num_episodes, max_steps)
        
        # Need to close and recreate environment to ensure identical conditions
        env.close()
        env = self.create_env(env_config)
        
        # Evaluate TE-DQN
        te_results = self.evaluate_model(self.te_dqn, "TE-DQN", env, num_episodes, max_steps)
        
        # Close environment
        env.close()
        
        # Save results
        scenario_dir = os.path.join(self.eval_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)
        
        with open(os.path.join(scenario_dir, "ff_dqn_results.json"), 'w') as f:
            json.dump(ff_results, f, indent=4)
        
        with open(os.path.join(scenario_dir, "te_dqn_results.json"), 'w') as f:
            json.dump(te_results, f, indent=4)
        
        # Generate comparison visualizations
        self.generate_comparison_plots(ff_results, te_results, scenario_name, scenario_dir)
        
        # Statistical comparison
        comparison = self.statistical_comparison(ff_results, te_results)
        with open(os.path.join(scenario_dir, "statistical_comparison.json"), 'w') as f:
            json.dump(comparison, f, indent=4)
        
        return {
            "ff_dqn": ff_results,
            "te_dqn": te_results,
            "comparison": comparison
        }
    
    def run_all_evaluations(self, num_episodes=10, max_steps=300):
        """Run evaluations on all scenarios"""
        all_results = {}
        
        for scenario_name in self.scenarios.keys():
            all_results[scenario_name] = self.evaluate_scenario(scenario_name, num_episodes, max_steps)
        
        # Create overall comparison
        self.generate_overall_comparison(all_results)
        
        return all_results
    
    def statistical_comparison(self, ff_results, te_results):
        """Perform statistical comparison between model results"""
        comparison = {
            "metrics": {},
            "p_values": {},
            "significant_difference": {}
        }
        
        # Metrics to compare
        metrics = [
            "rewards", 
            "completion_rates", 
            "rejection_rates", 
            "drop_rates", 
            "avg_latencies", 
            "energy_consumptions",
            "wake_decisions",
            "active_nodes_count"
        ]
        
        alpha = 0.05  # Significance level
        
        for metric in metrics:
            # Calculate t-test
            t_stat, p_value = stats.ttest_ind(ff_results[metric], te_results[metric])
            
            # Check if the difference is statistically significant
            significant = p_value < alpha
            
            # Determine which model performed better
            ff_mean = np.mean(ff_results[metric])
            te_mean = np.mean(te_results[metric])
            
            # For some metrics, lower is better
            if metric in ["rejection_rates", "drop_rates", "avg_latencies", "energy_consumptions"]:
                better_model = "TE-DQN" if te_mean < ff_mean else "FF-DQN"
            else:
                better_model = "TE-DQN" if te_mean > ff_mean else "FF-DQN"
            
            # Store results
            comparison["metrics"][metric] = {
                "ff_dqn_mean": float(ff_mean),
                "te_dqn_mean": float(te_mean),
                "difference": float(te_mean - ff_mean),
                "percent_difference": float((te_mean - ff_mean) / ff_mean * 100 if ff_mean != 0 else 0)
            }
            
            comparison["p_values"][metric] = float(p_value)
            comparison["significant_difference"][metric] = {
                "is_significant": bool(significant),
                "better_model": better_model if significant else "No significant difference"
            }
        
        return comparison
    
    def generate_comparison_plots(self, ff_results, te_results, scenario_name, output_dir):
        """Generate comparison visualizations"""
        # Set up the style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Reward Comparison
        plt.figure(figsize=(10, 6))
        plt.boxplot([ff_results["rewards"], te_results["rewards"]], 
                   labels=["FF-DQN", "TE-DQN"])
        plt.title(f"Reward Comparison - {scenario_name}")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "reward_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Metric Comparison Bar Chart
        metrics = {
            "Completion Rate": [np.mean(ff_results["completion_rates"]), np.mean(te_results["completion_rates"])],
            "Rejection Rate": [np.mean(ff_results["rejection_rates"]), np.mean(te_results["rejection_rates"])],
            "Avg Latency": [np.mean(ff_results["avg_latencies"]), np.mean(te_results["avg_latencies"])],
            "Energy Usage": [np.mean(ff_results["energy_consumptions"]), np.mean(te_results["energy_consumptions"])],
            "Wake Decisions": [np.mean(ff_results["wake_decisions"]), np.mean(te_results["wake_decisions"])]
        }
        
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, [m[0] for m in metrics.values()], width, label="FF-DQN")
        ax.bar(x + width/2, [m[1] for m in metrics.values()], width, label="TE-DQN")
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics.keys())
        ax.legend()
        ax.set_title(f"Performance Metrics Comparison - {scenario_name}")
        
        plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Energy-Latency Trade-off Scatter Plot
        plt.figure(figsize=(8, 8))
        
        # Get episode-wise data
        ff_latencies = []
        ff_energies = []
        te_latencies = []
        te_energies = []
        
        for ep_data in ff_results["episode_data"]:
            avg_latency = np.mean([step["avg_latency"] for step in ep_data["step_data"] if step["avg_latency"] > 0])
            avg_energy = np.mean([step["energy_consumption"] for step in ep_data["step_data"] if step["energy_consumption"] > 0])
            if not np.isnan(avg_latency) and not np.isnan(avg_energy):
                ff_latencies.append(avg_latency)
                ff_energies.append(avg_energy)
        
        for ep_data in te_results["episode_data"]:
            avg_latency = np.mean([step["avg_latency"] for step in ep_data["step_data"] if step["avg_latency"] > 0])
            avg_energy = np.mean([step["energy_consumption"] for step in ep_data["step_data"] if step["energy_consumption"] > 0])
            if not np.isnan(avg_latency) and not np.isnan(avg_energy):
                te_latencies.append(avg_latency)
                te_energies.append(avg_energy)
        
        plt.scatter(ff_latencies, ff_energies, label="FF-DQN", color="blue", alpha=0.7)
        plt.scatter(te_latencies, te_energies, label="TE-DQN", color="red", alpha=0.7)
        
        # Add centroids
        plt.scatter(np.mean(ff_latencies), np.mean(ff_energies), color="blue", s=100, marker="X", edgecolor="black", label="FF-DQN Mean")
        plt.scatter(np.mean(te_latencies), np.mean(te_energies), color="red", s=100, marker="X", edgecolor="black", label="TE-DQN Mean")
        
        plt.xlabel("Latency (s)")
        plt.ylabel("Energy Consumption (J)")
        plt.title(f"Energy-Latency Trade-off - {scenario_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "energy_latency_tradeoff.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Wake-up Action Analysis
        plt.figure(figsize=(10, 6))
        
        # Count wake actions by step
        ff_wake_by_step = defaultdict(int)
        te_wake_by_step = defaultdict(int)
        
        for ep_data in ff_results["episode_data"]:
            for step_data in ep_data["step_data"]:
                if step_data["is_wake_action"]:
                    ff_wake_by_step[step_data["step"]] += 1
        
        for ep_data in te_results["episode_data"]:
            for step_data in ep_data["step_data"]:
                if step_data["is_wake_action"]:
                    te_wake_by_step[step_data["step"]] += 1
        
        # Convert to lists for plotting
        steps = sorted(set(list(ff_wake_by_step.keys()) + list(te_wake_by_step.keys())))
        ff_wakes = [ff_wake_by_step.get(s, 0) for s in steps]
        te_wakes = [te_wake_by_step.get(s, 0) for s in steps]
        
        plt.plot(steps, ff_wakes, 'b-', label="FF-DQN")
        plt.plot(steps, te_wakes, 'r-', label="TE-DQN")
        plt.xlabel("Simulation Step")
        plt.ylabel("Number of Wake Actions")
        plt.title(f"Wake-up Action Distribution - {scenario_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "wakeup_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_overall_comparison(self, all_results):
        """Generate overall comparison across all scenarios"""
        # Create comparison table data
        scenarios = list(all_results.keys())
        metrics = ["avg_reward", "avg_completion_rate", "avg_latency", "avg_energy_consumption", "avg_wake_decisions"]
        
        # Extract data
        ff_data = {scenario: all_results[scenario]["ff_dqn"]["avg_metrics"] for scenario in scenarios}
        te_data = {scenario: all_results[scenario]["te_dqn"]["avg_metrics"] for scenario in scenarios}
        
        # 1. Generate summary table as JSON
        summary = {
            "ff_dqn": {metric: {scenario: ff_data[scenario][metric] for scenario in scenarios} for metric in metrics},
            "te_dqn": {metric: {scenario: te_data[scenario][metric] for scenario in scenarios} for metric in metrics},
            "comparison": {
                metric: {
                    scenario: {
                        "difference": te_data[scenario][metric] - ff_data[scenario][metric],
                        "percent_improvement": (te_data[scenario][metric] - ff_data[scenario][metric]) / ff_data[scenario][metric] * 100 if ff_data[scenario][metric] != 0 else 0
                    } for scenario in scenarios
                } for metric in metrics
            }
        }
        
        with open(os.path.join(self.eval_dir, "overall_comparison.json"), 'w') as f:
            json.dump(summary, f, indent=4)
        
        # 2. Generate radar plot for each scenario
        for scenario in scenarios:
            plt.figure(figsize=(10, 8))
            
            # Prepare data for radar plot
            # Normalize metrics to 0-1 range for radar plot
            ff_metrics = []
            te_metrics = []
            labels = []
            
            for metric in ["avg_completion_rate", "avg_reward"]:
                ff_val = ff_data[scenario][metric]
                te_val = te_data[scenario][metric]
                max_val = max(ff_val, te_val) * 1.1  # Add 10% for better visualization
                ff_metrics.append(ff_val / max_val if max_val != 0 else 0)
                te_metrics.append(te_val / max_val if max_val != 0 else 0)
                labels.append(metric.replace("avg_", ""))
            
            # For metrics where lower is better, invert the normalization
            for metric in ["avg_latency", "avg_energy_consumption"]:
                ff_val = ff_data[scenario][metric]
                te_val = te_data[scenario][metric]
                max_val = max(ff_val, te_val) * 1.1
                # Invert so lower values get higher scores on radar
                ff_metrics.append(1 - (ff_val / max_val) if max_val != 0 else 0)
                te_metrics.append(1 - (te_val / max_val) if max_val != 0 else 0)
                labels.append(metric.replace("avg_", "") + " (lower is better)")
            
            # Add wake decisions as a neutral metric
            metric = "avg_wake_decisions"
            ff_val = ff_data[scenario][metric]
            te_val = te_data[scenario][metric]
            max_val = max(ff_val, te_val) * 1.1
            ff_metrics.append(ff_val / max_val if max_val != 0 else 0)
            te_metrics.append(te_val / max_val if max_val != 0 else 0)
            labels.append("wake decisions")
            
            # Add one more point to close the polygon
            ff_metrics.append(ff_metrics[0])
            te_metrics.append(te_metrics[0])
            labels.append(labels[0])
            
            # Number of variables
            N = len(labels) - 1
            
            # Angle of each axis
            angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            # Create radar plot
            ax = plt.subplot(111, polar=True)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Draw polygon and points
            ax.plot(angles, ff_metrics, 'b-', linewidth=2, label='FF-DQN')
            ax.plot(angles, te_metrics, 'r-', linewidth=2, label='TE-DQN')
            ax.fill(angles, ff_metrics, 'b', alpha=0.1)
            ax.fill(angles, te_metrics, 'r', alpha=0.1)
            
            # Set labels
            plt.xticks(angles[:-1], labels[:-1])
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.title(f"Model Comparison - {scenario}")
            plt.tight_layout()
            plt.savefig(os.path.join(self.eval_dir, f"{scenario}_radar.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Generate bar charts comparing models across scenarios
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            
            metric_name = metric.replace("avg_", "").replace("_", " ").title()
            ff_values = [ff_data[s][metric] for s in scenarios]
            te_values = [te_data[s][metric] for s in scenarios]
            
            x = np.arange(len(scenarios))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - width/2, ff_values, width, label='FF-DQN')
            ax.bar(x + width/2, te_values, width, label='TE-DQN')
            
            ax.set_xlabel('Scenario')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Comparison Across Scenarios')
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.eval_dir, f"{metric}_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Generate combined radar plot for all scenarios
        # This is too complex for a single plot, so we'll create a grid of radar plots
        fig, axs = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'polar': True})
        axs = axs.flatten()
        
        for i, scenario in enumerate(scenarios):
            if i >= len(axs):
                break  # In case we have more scenarios than subplots
                
            ax = axs[i]
            
            # Prepare data for radar plot - similar to individual radar plots
            ff_metrics = []
            te_metrics = []
            labels = []
            
            for metric in ["avg_completion_rate", "avg_reward"]:
                ff_val = ff_data[scenario][metric]
                te_val = te_data[scenario][metric]
                max_val = max(ff_val, te_val) * 1.1
                ff_metrics.append(ff_val / max_val if max_val != 0 else 0)
                te_metrics.append(te_val / max_val if max_val != 0 else 0)
                labels.append(metric.replace("avg_", ""))
            
            for metric in ["avg_latency", "avg_energy_consumption"]:
                ff_val = ff_data[scenario][metric]
                te_val = te_data[scenario][metric]
                max_val = max(ff_val, te_val) * 1.1
                ff_metrics.append(1 - (ff_val / max_val) if max_val != 0 else 0)
                te_metrics.append(1 - (te_val / max_val) if max_val != 0 else 0)
                labels.append(metric.replace("avg_", ""))
            
            metric = "avg_wake_decisions"
            ff_val = ff_data[scenario][metric]
            te_val = te_data[scenario][metric]
            max_val = max(ff_val, te_val) * 1.1
            ff_metrics.append(ff_val / max_val if max_val != 0 else 0)
            te_metrics.append(te_val / max_val if max_val != 0 else 0)
            labels.append("wake")
            
            # Close the polygon
            ff_metrics.append(ff_metrics[0])
            te_metrics.append(te_metrics[0])
            labels.append(labels[0])
            
            # Number of variables
            N = len(labels) - 1
            
            # Angle of each axis
            angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            # Configure axis
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Draw polygon and points
            ax.plot(angles, ff_metrics, 'b-', linewidth=2, label='FF-DQN')
            ax.plot(angles, te_metrics, 'r-', linewidth=2, label='TE-DQN')
            ax.fill(angles, ff_metrics, 'b', alpha=0.1)
            ax.fill(angles, te_metrics, 'r', alpha=0.1)
            
            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels[:-1], fontsize=8)
            ax.set_title(scenario)
            
            # Only add legend to the first subplot
            if i == 0:
                ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Handle any unused subplots
        for i in range(len(scenarios), len(axs)):
            axs[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, "all_scenarios_radar.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Energy-Latency trade-off comparison across scenarios
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot with one point per scenario per model
        for scenario in scenarios:
            ff_latency = ff_data[scenario]["avg_latency"]
            ff_energy = ff_data[scenario]["avg_energy_consumption"]
            te_latency = te_data[scenario]["avg_latency"]
            te_energy = te_data[scenario]["avg_energy_consumption"]
            
            plt.scatter(ff_latency, ff_energy, marker='o', s=100, label=f'FF-DQN: {scenario}')
            plt.scatter(te_latency, te_energy, marker='x', s=100, label=f'TE-DQN: {scenario}')
            
            # Draw arrow from FF-DQN to TE-DQN to show improvement direction
            plt.arrow(ff_latency, ff_energy, te_latency - ff_latency, te_energy - ff_energy,
                    head_width=0.05, head_length=0.1, fc='black', ec='black', alpha=0.3)
        
        plt.xlabel('Average Latency (s)')
        plt.ylabel('Average Energy Consumption (J)')
        plt.title('Energy-Latency Trade-off Across Scenarios')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, "energy_latency_tradeoff_all.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Improvement percentage across scenarios
        plt.figure(figsize=(12, 8))
        
        # Calculate improvement percentages
        improvements = {}
        for metric in ["avg_completion_rate", "avg_reward"]:
            improvements[metric] = []
            for scenario in scenarios:
                ff_val = ff_data[scenario][metric]
                te_val = te_data[scenario][metric]
                if ff_val != 0:
                    imp = (te_val - ff_val) / ff_val * 100
                    improvements[metric].append(imp)
                else:
                    improvements[metric].append(0)
        
        # For metrics where lower is better, invert the calculation
        for metric in ["avg_latency", "avg_energy_consumption"]:
            improvements[metric] = []
            for scenario in scenarios:
                ff_val = ff_data[scenario][metric]
                te_val = te_data[scenario][metric]
                if ff_val != 0:
                    # Negative values mean TE-DQN is better (lower values)
                    imp = -((te_val - ff_val) / ff_val * 100)
                    improvements[metric].append(imp)
                else:
                    improvements[metric].append(0)
        
        # Define colors for positive and negative values
        colors = []
        for metric in improvements:
            metric_colors = []
            for val in improvements[metric]:
                if val >= 0:
                    metric_colors.append('green')
                else:
                    metric_colors.append('red')
            colors.append(metric_colors)
        
        # Plot
        fig, axs = plt.subplots(len(improvements), 1, figsize=(12, 3*len(improvements)))
        
        for i, (metric, values) in enumerate(improvements.items()):
            ax = axs[i]
            bars = ax.bar(scenarios, values, color=colors[i])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_ylabel('% Improvement')
            ax.set_title(f'TE-DQN vs FF-DQN: {metric.replace("avg_", "")}')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, "percentage_improvements.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Wake decision timing analysis
        plt.figure(figsize=(10, 6))
        
        # Calculate cumulative wake decisions over time for each model across all scenarios
        ff_cumulative_wakes = defaultdict(int)
        te_cumulative_wakes = defaultdict(int)
        
        for scenario in scenarios:
            # Get episode data
            ff_scenario_data = all_results[scenario]["ff_dqn"]["episode_data"]
            te_scenario_data = all_results[scenario]["te_dqn"]["episode_data"]
            
            # Count wake actions by step
            for ep_data in ff_scenario_data:
                for step_data in ep_data["step_data"]:
                    if step_data["is_wake_action"]:
                        ff_cumulative_wakes[step_data["step"]] += 1
            
            for ep_data in te_scenario_data:
                for step_data in ep_data["step_data"]:
                    if step_data["is_wake_action"]:
                        te_cumulative_wakes[step_data["step"]] += 1
        
        # Convert to cumulative distribution
        steps = sorted(set(list(ff_cumulative_wakes.keys()) + list(te_cumulative_wakes.keys())))
        ff_cum_values = []
        te_cum_values = []
        
        ff_total = 0
        te_total = 0
        for step in steps:
            ff_total += ff_cumulative_wakes.get(step, 0)
            te_total += te_cumulative_wakes.get(step, 0)
            ff_cum_values.append(ff_total)
            te_cum_values.append(te_total)
        
        # Normalize to percentages
        if ff_total > 0:
            ff_cum_values = [v / ff_total * 100 for v in ff_cum_values]
        if te_total > 0:
            te_cum_values = [v / te_total * 100 for v in te_cum_values]
        
        plt.plot(steps, ff_cum_values, 'b-', label="FF-DQN")
        plt.plot(steps, te_cum_values, 'r-', label="TE-DQN")
        plt.xlabel("Simulation Step")
        plt.ylabel("Cumulative Wake Actions (%)")
        plt.title("Wake-up Action Timing Analysis")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.eval_dir, "wakeup_timing_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
                
        # 8. Generate HTML report
        self._generate_html_report()
    
    def _generate_html_report(self):
        """Generate an HTML report of all evaluation results"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>VEC Model Evaluation Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .summary {{ margin-bottom: 30px; }}
                .scenario {{ margin-bottom: 50px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
                .scenario-title {{ color: #0066cc; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metrics-table th {{ padding-top: 12px; padding-bottom: 12px; background-color: #0066cc; color: white; }}
                .better {{ color: green; font-weight: bold; }}
                .worse {{ color: red; }}
                .plots {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
                .plot {{ margin-bottom: 20px; }}
                .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>VEC Model Evaluation Results</h1>
                <p>Evaluation Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="summary">
                    <h2>Overall Comparison</h2>
                    <p>This report compares the performance of FF-DQN and TE-DQN models across multiple scenarios.</p>
                    
                    <div class="plots">
                        <div class="plot">
                            <h3>Energy-Latency Trade-off Across Scenarios</h3>
                            <img src="energy_latency_tradeoff_all.png" alt="Energy-Latency Trade-off">
                        </div>
                        <div class="plot">
                            <h3>Improvement Percentages</h3>
                            <img src="percentage_improvements.png" alt="Improvement Percentages">
                        </div>
                        <div class="plot">
                            <h3>Wake-up Action Timing Analysis</h3>
                            <img src="wakeup_timing_analysis.png" alt="Wake-up Timing Analysis">
                        </div>
                        <div class="plot">
                            <h3>All Scenarios Radar Comparison</h3>
                            <img src="all_scenarios_radar.png" alt="All Scenarios Radar">
                        </div>
                    </div>
                </div>
        """
        
        # Add individual scenario sections
        for scenario in self.scenarios:
            html_content += f"""
                <div class="scenario">
                    <h2 class="scenario-title">{self.scenarios[scenario]['name']}</h2>
                    <p>{self.scenarios[scenario]['description']}</p>
                    
                    <h3>Performance Metrics</h3>
                    <div class="plots">
                        <div class="plot">
                            <h4>Reward Comparison</h4>
                            <img src="{scenario}/reward_comparison.png" alt="Reward Comparison">
                        </div>
                        <div class="plot">
                            <h4>Metrics Comparison</h4>
                            <img src="{scenario}/metrics_comparison.png" alt="Metrics Comparison">
                        </div>
                        <div class="plot">
                            <h4>Energy-Latency Trade-off</h4>
                            <img src="{scenario}/energy_latency_tradeoff.png" alt="Energy-Latency Trade-off">
                        </div>
                        <div class="plot">
                            <h4>Wake-up Distribution</h4>
                            <img src="{scenario}/wakeup_distribution.png" alt="Wake-up Distribution">
                        </div>
                    </div>
                    
                    <h3>Radar Plot</h3>
                    <div class="plots">
                        <div class="plot">
                            <img src="{scenario}_radar.png" alt="{scenario} Radar Plot">
                        </div>
                    </div>
                </div>
            """
        
        # Close HTML
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(os.path.join(self.eval_dir, "evaluation_report.html"), 'w') as f:
            f.write(html_content)
        
        print(f"HTML report generated at {os.path.join(self.eval_dir, 'evaluation_report.html')}")


def main():
    """Main function to run evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate and compare VEC offloading models')
    parser.add_argument('--ff_dqn_model', type=str, required=True, help='Path to FF-DQN model file')
    parser.add_argument('--te_dqn_model', type=str, required=True, help='Path to TE-DQN model file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory for results')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes per scenario')
    parser.add_argument('--steps', type=int, default=300, help='Maximum steps per episode')
    parser.add_argument('--scenario', type=str, default=None, help='Specific scenario to evaluate (optional)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.output_dir)
    
    # Load models
    evaluator.load_models(args.ff_dqn_model, args.te_dqn_model)
    
    # Run evaluations
    if args.scenario is not None:
        if args.scenario in evaluator.scenarios:
            print(f"Evaluating scenario: {args.scenario}")
            evaluator.evaluate_scenario(args.scenario, args.episodes, args.steps)
        else:
            print(f"Scenario {args.scenario} not found. Available scenarios:")
            for scenario_name, scenario_data in evaluator.scenarios.items():
                print(f"  - {scenario_name}: {scenario_data['name']}")
    else:
        print("Running all evaluations")
        evaluator.run_all_evaluations(args.episodes, args.steps)
    
    print(f"Evaluation complete. Results saved to {evaluator.eval_dir}")


if __name__ == "__main__":
    main()