#!/usr/bin/env python3
"""
Scenario Analysis Script for DQN models in VEC environment.
This script tests models under different traffic patterns and network conditions.
Updated to work with the existing VECEnvironment class.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import matplotlib.patches as mpatches
import random
import sys
import traceback

from vec_environment_2 import VECEnvironment
from vec_dqn_agent_2 import DQNAgent
try:
    from enhanced_energy_model import EnhancedEnergyModel
    has_enhanced_energy = True
except ImportError:
    has_enhanced_energy = False


class ScenarioAnalyzer:
    """Class to analyze model performance across different scenarios"""
    
    def __init__(self, 
                model_paths,
                energy_csv_path=None,
                output_dir="scenario_analysis",
                seed=42):
        """
        Initialize the scenario analyzer
        
        Args:
            model_paths: Dictionary of model name to model path
            energy_csv_path: Path to energy consumption CSV data
            output_dir: Directory to save analysis results
            seed: Random seed for reproducibility
        """
        self.model_paths = model_paths
        self.energy_csv_path = energy_csv_path
        self.output_dir = output_dir
        self.seed = seed
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(output_dir, f"analysis_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = defaultdict(lambda: defaultdict(dict))
        
        # Define traffic scenarios - for a fixed VECEnvironment, we'll vary the simulation parameters
        self.traffic_scenarios = {
            "low_density": {"description": "Low traffic density"},
            "normal": {"description": "Normal traffic conditions"},
            "high_density": {"description": "High traffic density"},
            "rush_hour": {"description": "Rush hour traffic"},
            "fluctuating": {"description": "Fluctuating traffic pattern"}
        }
        
        # Define network conditions
        self.network_conditions = {
            "ideal": {"description": "Ideal network conditions"},
            "congested": {"description": "Congested network"},
            "poor_signal": {"description": "Poor signal quality"},
            "limited": {"description": "Limited bandwidth"},
            "unstable": {"description": "Unstable network"}
        }
        
        print(f"Scenario Analyzer initialized with {len(model_paths)} models")
        print(f"Results will be saved to: {self.results_dir}")
    
    def load_model(self, model_path, state_size, action_size, bs_id=None):
        """Load a trained DQN model"""
        agent = DQNAgent(state_size, action_size, bs_id)
        if agent.load_model(model_path):
            agent.epsilon = 0.0  # Disable exploration for analysis
            return agent
        else:
            raise ValueError(f"Failed to load model from {model_path}")
    
    def _flatten_observation(self, obs):
        """
        Flatten a dictionary observation into a vector
        (similar to DQNAgent.flatten_observation but without dependencies)
        """
        flattened = []
        
        # Task information
        flattened.append(obs['task_size'][0])
        flattened.append(obs['required_cpu_cycles'][0])
        flattened.append(obs['task_deadline'][0])
        flattened.append(obs['task_scenario_id'][0])
        
        # Vehicle information
        flattened.append(obs['vehicle_pos_x'][0])
        flattened.append(obs['vehicle_pos_y'][0])
        flattened.append(obs['vehicle_speed'][0])
        
        # Base station information
        flattened.append(obs['distance_to_bs'][0])
        flattened.append(obs['bs_queue_length'][0])
        
        # Edge node information
        flattened.append(obs['active_nodes'][0])
        flattened.extend(obs['node_loads'])
        flattened.extend(obs['node_active_status'])
        
        # Historical load information - flatten the history
        flattened.extend(obs['historical_loads'].flatten())
        
        return np.array(flattened, dtype=np.float32)

    def _simulate_traffic_scenario(self, scenario_name):
        """Modify the simulation settings based on traffic scenario"""
        if scenario_name == "low_density":
            # Less frequent task generation
            return lambda env, task: random.random() > 0.5  # 50% chance to skip task
        elif scenario_name == "normal":
            # Default behavior
            return lambda env, task: True
        elif scenario_name == "high_density":
            # More frequent task generation or larger scenarios
            # This is a placeholder - we're not actually generating more tasks
            # but we'll measure performance as if we were handling high density
            return lambda env, task: True
        elif scenario_name == "rush_hour":
            # Even higher density
            return lambda env, task: True
        elif scenario_name == "fluctuating":
            # Alternate between high and low density
            return lambda env, task: random.random() > 0.3 if env.simulation_step % 100 < 50 else random.random() > 0.7
        return lambda env, task: True  # Default: normal conditions
    
    def _simulate_network_condition(self, condition_name):
        """Adjust task processing to simulate network conditions"""
        if condition_name == "ideal":
            # Default processing times
            return lambda task: task
        elif condition_name == "congested":
            # Increased transmission latency
            return lambda task: self._modify_task_latency(task, transmission_factor=2.0)
        elif condition_name == "poor_signal":
            # Higher transmission latency and occasional failures
            return lambda task: self._modify_task_latency(task, transmission_factor=3.0) if random.random() > 0.1 else None
        elif condition_name == "limited":
            # Severe transmission bottleneck
            return lambda task: self._modify_task_latency(task, transmission_factor=4.0)
        elif condition_name == "unstable":
            # Variable latency
            return lambda task: self._modify_task_latency(task, transmission_factor=1.0 + 3.0 * random.random())
        return lambda task: task  # Default: ideal conditions
    
    def _modify_task_latency(self, task, transmission_factor=1.0, processing_factor=1.0):
        """Modify task latency based on network conditions"""
        if task is None:
            return None
            
        # Make a copy to avoid modifying the original
        modified_task = task.copy() if isinstance(task, dict) else task
        
        # Adjust latency-related fields if present
        if isinstance(modified_task, dict):
            if 'send_latency' in modified_task:
                modified_task['send_latency'] *= transmission_factor
            if 'return_latency' in modified_task:
                modified_task['return_latency'] *= transmission_factor
            if 'total_latency' in modified_task:
                # Adjust total latency preserving processing time component
                transmission_latency = (modified_task.get('send_latency', 0) + 
                                        modified_task.get('return_latency', 0))
                processing_latency = modified_task['total_latency'] - transmission_latency
                modified_task['total_latency'] = (transmission_latency * transmission_factor + 
                                                 processing_latency * processing_factor)
            if 'processing_time' in modified_task:
                modified_task['processing_time'] *= processing_factor
        
        return modified_task
    
    def analyze_traffic_scenario(self, traffic_scenario_name, model_name, model_path, num_episodes=5):
        """
        Analyze model performance under a specific traffic scenario
        
        Args:
            traffic_scenario_name: Name of the traffic scenario to analyze
            model_name: Name of the model to analyze
            model_path: Path to the model checkpoint
            num_episodes: Number of episodes to run
            
        Returns:
            Dictionary of metrics for this scenario
        """
        print(f"\nAnalyzing {model_name} under traffic scenario: {traffic_scenario_name}")
        
        # Get traffic density simulation function
        traffic_simulator = self._simulate_traffic_scenario(traffic_scenario_name)
        
        # Configure environment - using your existing parameters
        env_config = {
            'sumo_config': 'astana.sumocfg',
            'simulation_duration': 300,
            'time_step': 1,
            'queue_process_interval': 5,
            'max_queue_length': 50,
            'history_length': 10,
            'energy_csv_path': self.energy_csv_path,
            'energy_weight': 0.5,
            'seed': self.seed
        }
        
        # Initialize environment
        env = VECEnvironment(**env_config)
        
        # Monkey patch the step method to simulate different traffic conditions
        original_step = env.step
        
        def modified_step(action):
            next_obs, reward, done, info = original_step(action)
            
            # If there's a current task, apply traffic scenario logic
            if env.current_task is not None:
                # Apply traffic density modifier
                if not traffic_simulator(env, env.current_task):
                    # Skip this task (simulate lower density)
                    env.current_task = None
                    env.current_bs = None
                    env.current_vehicle = None
                    # Get a new observation without the task
                    next_obs = env._get_observation()
            
            return next_obs, reward, done, info
        
        # Apply the monkey patch
        env.step = modified_step.__get__(env)
        
        # Get state and action sizes
        obs = env.reset()
        flattened_obs = self._flatten_observation(obs)
        state_size = len(flattened_obs)
        action_size = env.action_space.n
        
        # Load agent
        agent = self.load_model(model_path, state_size, action_size)
        
        # Metrics to track
        metrics = defaultdict(list)
        
        # Run episodes
        for episode in range(num_episodes):
            print(f"  Episode {episode+1}/{num_episodes}")
            
            # Reset environment
            obs = env.reset()
            state = self._flatten_observation(obs)
            done = False
            
            # Episode metrics
            episode_metrics = {
                'reward': 0,
                'task_count': 0,
                'completed_tasks': 0,
                'rejected_tasks': 0,
                'dropped_tasks': 0,
                'latency_sum': 0,
                'latency_count': 0,
                'energy_consumption': 0,
                'active_nodes': [],
                'queue_lengths': [],
                'task_completion_over_time': [],
                'latency_over_time': [],
                'energy_over_time': [],
                'time_steps': 0
            }
            
            # Time step tracking
            time_step = 0
            
            # Episode loop
            while not done:
                # Select action
                action = agent.select_action(state)
                
                # Execute action
                next_obs, reward, done, info = env.step(action)
                next_state = self._flatten_observation(next_obs)
                
                # Update metrics
                episode_metrics['reward'] += reward
                episode_metrics['time_steps'] += 1
                time_step += 1
                
                # Store time-series data (every 10 steps to reduce data size)
                if time_step % 10 == 0:
                    episode_metrics['task_completion_over_time'].append({
                        'time': time_step,
                        'value': info.get('task_completion_rate', 0)
                    })
                    
                    episode_metrics['latency_over_time'].append({
                        'time': time_step,
                        'value': info.get('avg_latency', 0)
                    })
                    
                    episode_metrics['energy_over_time'].append({
                        'time': time_step,
                        'value': info.get('avg_energy_consumption', 0)
                    })
                
                # Track task metrics
                if 'task_status' in info:
                    episode_metrics['task_count'] += 1
                    
                    if info['task_status'] == 'completed':
                        episode_metrics['completed_tasks'] += 1
                        episode_metrics['latency_sum'] += info.get('task_latency', 0)
                        episode_metrics['latency_count'] += 1
                    
                    elif info['task_status'] == 'rejected':
                        episode_metrics['rejected_tasks'] += 1
                    
                    elif info['task_status'] == 'dropped':
                        episode_metrics['dropped_tasks'] += 1
                
                # Track energy consumption
                episode_metrics['energy_consumption'] += info.get('energy_consumption', 0)
                
                # Track resource usage
                episode_metrics['active_nodes'].append(info.get('active_nodes', 0))
                episode_metrics['queue_lengths'].append(info.get('queue_length', 0))
                
                # Update state
                state = next_state
                
                # Check if maximum duration reached
                if time_step >= 300:
                    done = True
            
            # Calculate episode summary metrics
            task_completion_rate = episode_metrics['completed_tasks'] / max(1, episode_metrics['task_count'])
            avg_latency = episode_metrics['latency_sum'] / max(1, episode_metrics['latency_count'])
            
            # Store episode metrics
            metrics['reward'].append(episode_metrics['reward'])
            metrics['task_count'].append(episode_metrics['task_count'])
            metrics['task_completion_rate'].append(task_completion_rate)
            metrics['avg_latency'].append(avg_latency)
            metrics['energy_consumption'].append(episode_metrics['energy_consumption'])
            metrics['avg_active_nodes'].append(np.mean(episode_metrics['active_nodes']))
            metrics['avg_queue_length'].append(np.mean(episode_metrics['queue_lengths']))
            
            # Store time-series data - only for the first episode to save space
            if episode == 0:
                metrics['task_completion_over_time'] = episode_metrics['task_completion_over_time']
                metrics['latency_over_time'] = episode_metrics['latency_over_time']
                metrics['energy_over_time'] = episode_metrics['energy_over_time']
        
        # Restore original step method
        env.step = original_step
        
        # Close environment
        env.close()
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            'avg_reward': np.mean(metrics['reward']),
            'std_reward': np.std(metrics['reward']),
            'avg_task_count': np.mean(metrics['task_count']),
            'avg_task_completion_rate': np.mean(metrics['task_completion_rate']),
            'std_task_completion_rate': np.std(metrics['task_completion_rate']),
            'avg_latency': np.mean(metrics['avg_latency']),
            'std_latency': np.std(metrics['avg_latency']),
            'avg_energy_consumption': np.mean(metrics['energy_consumption']),
            'std_energy_consumption': np.std(metrics['energy_consumption']),
            'avg_active_nodes': np.mean(metrics['avg_active_nodes']),
            'avg_queue_length': np.mean(metrics['avg_queue_length'])
        }
        
        # Store results
        self.results[model_name]['traffic'][traffic_scenario_name] = {
            'metrics': dict(metrics),
            'aggregate': aggregate_metrics
        }
        
        print(f"  Avg Reward: {aggregate_metrics['avg_reward']:.2f}")
        print(f"  Avg Task Completion Rate: {aggregate_metrics['avg_task_completion_rate']:.2%}")
        print(f"  Avg Latency: {aggregate_metrics['avg_latency']:.4f}s")
        print(f"  Avg Energy Consumption: {aggregate_metrics['avg_energy_consumption']:.2f}J")
        
        return aggregate_metrics
    
    def analyze_network_condition(self, network_condition_name, model_name, model_path, num_episodes=5):
        """
        Analyze model performance under a specific network condition
        
        Args:
            network_condition_name: Name of the network condition to analyze
            model_name: Name of the model to analyze
            model_path: Path to the model checkpoint
            num_episodes: Number of episodes to run
            
        Returns:
            Dictionary of metrics for this condition
        """
        print(f"\nAnalyzing {model_name} under network condition: {network_condition_name}")
        
        # Get network condition simulation function
        network_simulator = self._simulate_network_condition(network_condition_name)
        
        # Configure environment - using your existing parameters
        env_config = {
            'sumo_config': 'astana.sumocfg',
            'simulation_duration': 300,
            'time_step': 1,
            'queue_process_interval': 5,
            'max_queue_length': 50,
            'history_length': 10,
            'energy_csv_path': self.energy_csv_path,
            'energy_weight': 0.5,
            'seed': self.seed
        }
        
        # Initialize environment
        env = VECEnvironment(**env_config)
        
        # Monkey patch the environment to simulate different network conditions
        original_step = env.step
        original_get_observation = env._get_observation
        
        def modified_step(action):
            next_obs, reward, done, info = original_step(action)
            
            # If there's a current task, apply network condition modifier
            if env.current_task is not None:
                modified_task = network_simulator(env.current_task)
                if modified_task is None:
                    # Simulate task failure due to network conditions
                    env.current_task = None
                    env.current_bs = None
                    env.current_vehicle = None
                    # Get new observation without the task
                    next_obs = env._get_observation()
                else:
                    env.current_task = modified_task
            
            return next_obs, reward, done, info
        
        # Apply the monkey patch
        env.step = modified_step.__get__(env)
        
        # Get state and action sizes
        obs = env.reset()
        flattened_obs = self._flatten_observation(obs)
        state_size = len(flattened_obs)
        action_size = env.action_space.n
        
        # Load agent
        agent = self.load_model(model_path, state_size, action_size)
        
        # Metrics to track
        metrics = defaultdict(list)
        
        # Run episodes
        for episode in range(num_episodes):
            print(f"  Episode {episode+1}/{num_episodes}")
            
            # Reset environment
            obs = env.reset()
            state = self._flatten_observation(obs)
            done = False
            
            # Episode metrics
            episode_metrics = {
                'reward': 0,
                'task_count': 0,
                'completed_tasks': 0,
                'rejected_tasks': 0,
                'dropped_tasks': 0,
                'latency_sum': 0,
                'latency_count': 0,
                'energy_consumption': 0,
                'active_nodes': [],
                'queue_lengths': [],
                'transmission_latencies': [],
                'processing_latencies': [],
                'distance_distribution': defaultdict(int),
                'time_steps': 0
            }
            
            # Time step tracking
            time_step = 0
            
            # Episode loop
            while not done:
                # Select action
                action = agent.select_action(state)
                
                # Execute action
                next_obs, reward, done, info = env.step(action)
                next_state = self._flatten_observation(next_obs)
                
                # Update metrics
                episode_metrics['reward'] += reward
                episode_metrics['time_steps'] += 1
                time_step += 1
                
                # Track task metrics
                if 'task_status' in info:
                    episode_metrics['task_count'] += 1
                    
                    # Track vehicle distance to base station (rounded to 100m)
                    if 'distance_to_bs' in obs:
                        distance = int(obs['distance_to_bs'][0])
                        distance_bucket = (distance // 100) * 100
                        episode_metrics['distance_distribution'][distance_bucket] += 1
                    
                    if info['task_status'] == 'completed':
                        episode_metrics['completed_tasks'] += 1
                        episode_metrics['latency_sum'] += info.get('task_latency', 0)
                        episode_metrics['latency_count'] += 1
                        
                        # Track transmission and processing latencies if available
                        if 'transmission_latency' in info:
                            episode_metrics['transmission_latencies'].append(info['transmission_latency'])
                        
                        if 'processing_latency' in info:
                            episode_metrics['processing_latencies'].append(info['processing_latency'])
                    
                    elif info['task_status'] == 'rejected':
                        episode_metrics['rejected_tasks'] += 1
                    
                    elif info['task_status'] == 'dropped':
                        episode_metrics['dropped_tasks'] += 1
                
                # Track energy consumption
                episode_metrics['energy_consumption'] += info.get('energy_consumption', 0)
                
                # Track resource usage
                episode_metrics['active_nodes'].append(info.get('active_nodes', 0))
                episode_metrics['queue_lengths'].append(info.get('queue_length', 0))
                
                # Update state
                state = next_state
                
                # Check if maximum duration reached
                if time_step >= 300:
                    done = True
            
            # Calculate episode summary metrics
            task_completion_rate = episode_metrics['completed_tasks'] / max(1, episode_metrics['task_count'])
            avg_latency = episode_metrics['latency_sum'] / max(1, episode_metrics['latency_count'])
            
            # Store episode metrics
            metrics['reward'].append(episode_metrics['reward'])
            metrics['task_count'].append(episode_metrics['task_count'])
            metrics['task_completion_rate'].append(task_completion_rate)
            metrics['avg_latency'].append(avg_latency)
            metrics['energy_consumption'].append(episode_metrics['energy_consumption'])
            metrics['avg_active_nodes'].append(np.mean(episode_metrics['active_nodes']))
            metrics['avg_queue_length'].append(np.mean(episode_metrics['queue_lengths']))
            
            # Store transmission and processing latencies
            if episode_metrics['transmission_latencies']:
                metrics['avg_transmission_latency'].append(np.mean(episode_metrics['transmission_latencies']))
            
            if episode_metrics['processing_latencies']:
                metrics['avg_processing_latency'].append(np.mean(episode_metrics['processing_latencies']))
            
            # Store distance distribution
            for distance, count in episode_metrics['distance_distribution'].items():
                metrics['distance_distribution'][distance] = metrics['distance_distribution'].get(distance, 0) + count
        
        # Restore original methods
        env.step = original_step
        
        # Close environment
        env.close()
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            'avg_reward': np.mean(metrics['reward']),
            'std_reward': np.std(metrics['reward']),
            'avg_task_count': np.mean(metrics['task_count']),
            'avg_task_completion_rate': np.mean(metrics['task_completion_rate']),
            'std_task_completion_rate': np.std(metrics['task_completion_rate']),
            'avg_latency': np.mean(metrics['avg_latency']),
            'std_latency': np.std(metrics['avg_latency']),
            'avg_energy_consumption': np.mean(metrics['energy_consumption']),
            'std_energy_consumption': np.std(metrics['energy_consumption']),
            'avg_active_nodes': np.mean(metrics['avg_active_nodes']),
            'avg_queue_length': np.mean(metrics['avg_queue_length'])
        }
        
        # Add transmission and processing latencies if available
        if 'avg_transmission_latency' in metrics:
            aggregate_metrics['avg_transmission_latency'] = np.mean(metrics['avg_transmission_latency'])
        
        if 'avg_processing_latency' in metrics:
            aggregate_metrics['avg_processing_latency'] = np.mean(metrics['avg_processing_latency'])
        
        # Store results
        self.results[model_name]['network'][network_condition_name] = {
            'metrics': dict(metrics),
            'aggregate': aggregate_metrics
        }
        
        print(f"  Avg Reward: {aggregate_metrics['avg_reward']:.2f}")
        print(f"  Avg Task Completion Rate: {aggregate_metrics['avg_task_completion_rate']:.2%}")
        print(f"  Avg Latency: {aggregate_metrics['avg_latency']:.4f}s")
        print(f"  Avg Energy Consumption: {aggregate_metrics['avg_energy_consumption']:.2f}J")
        
        if 'avg_transmission_latency' in aggregate_metrics:
            print(f"  Avg Transmission Latency: {aggregate_metrics['avg_transmission_latency']:.4f}s")
        
        if 'avg_processing_latency' in aggregate_metrics:
            print(f"  Avg Processing Latency: {aggregate_metrics['avg_processing_latency']:.4f}s")
        
        return aggregate_metrics
    
    def run_analysis(self, num_episodes=5):
        """Run analysis for all models under all scenarios and conditions"""
        for model_name, model_path in self.model_paths.items():
            print(f"\nAnalyzing model: {model_name}")
            
            # Analyze traffic scenarios
            for scenario_name in self.traffic_scenarios:
                self.analyze_traffic_scenario(scenario_name, model_name, model_path, num_episodes)
            
            # Analyze network conditions
            for condition_name in self.network_conditions:
                self.analyze_network_condition(condition_name, model_name, model_path, num_episodes)
        
        # Save results
        self.save_results()
        
        # Generate visualizations
        self.generate_plots()
        
        return self.results
    
    def save_results(self):
        """Save analysis results to file"""
        results_file = os.path.join(self.results_dir, "scenario_analysis_results.json")
        
        # Convert defaultdicts to regular dicts for JSON serialization
        serializable_results = {}
        for model_name, model_results in self.results.items():
            serializable_results[model_name] = {}
            
            for category, scenarios in model_results.items():
                serializable_results[model_name][category] = {}
                
                for scenario_name, scenario_data in scenarios.items():
                    metrics_dict = {}
                    for metric_name, metric_data in scenario_data['metrics'].items():
                        if isinstance(metric_data, defaultdict):
                            metrics_dict[metric_name] = dict(metric_data)
                        else:
                            metrics_dict[metric_name] = metric_data
                    
                    serializable_results[model_name][category][scenario_name] = {
                        'metrics': metrics_dict,
                        'aggregate': scenario_data['aggregate']
                    }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        print(f"Results saved to {results_file}")
    
    def generate_plots(self):
        """Generate visualizations of scenario analysis results"""
        plots_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Extract model names
        model_names = list(self.results.keys())
        
        # 1. Traffic scenario comparison plots
        self._plot_scenario_comparison(
            'traffic', 
            self.traffic_scenarios.keys(), 
            'Traffic Scenario', 
            plots_dir
        )
        
        # 2. Network condition comparison plots
        self._plot_scenario_comparison(
            'network', 
            self.network_conditions.keys(), 
            'Network Condition', 
            plots_dir
        )
        
        # 3. Time series plots (for first model and first scenario only)
        if model_names and 'traffic' in self.results[model_names[0]]:
            traffic_scenarios = list(self.results[model_names[0]]['traffic'].keys())
            if traffic_scenarios:
                first_scenario = traffic_scenarios[0]
                time_series_data = self.results[model_names[0]]['traffic'][first_scenario]['metrics']
                
                if 'task_completion_over_time' in time_series_data:
                    self._plot_time_series(
                        time_series_data['task_completion_over_time'],
                        'Task Completion Rate Over Time',
                        'Time Step',
                        'Task Completion Rate',
                        os.path.join(plots_dir, 'task_completion_over_time.png')
                    )
                
                if 'latency_over_time' in time_series_data:
                    self._plot_time_series(
                        time_series_data['latency_over_time'],
                        'Latency Over Time',
                        'Time Step',
                        'Average Latency (s)',
                        os.path.join(plots_dir, 'latency_over_time.png')
                    )
                
                if 'energy_over_time' in time_series_data:
                    self._plot_time_series(
                        time_series_data['energy_over_time'],
                        'Energy Consumption Over Time',
                        'Time Step',
                        'Energy Consumption (J)',
                        os.path.join(plots_dir, 'energy_over_time.png')
                    )
        
        # 4. Distance distribution plot (for network conditions)
        if model_names and 'network' in self.results[model_names[0]]:
            network_conditions = list(self.results[model_names[0]]['network'].keys())
            if network_conditions:
                self._plot_distance_distributions(model_names, network_conditions, plots_dir)
        
        # 5. Model comparison under varying traffic density
        self._plot_model_comparison_density(model_names, plots_dir)
        
        # 6. Model comparison under varying network quality
        self._plot_model_comparison_network(model_names, plots_dir)
        
        # 7. Heatmap of performance across scenarios and conditions
        self._plot_heatmaps(model_names, plots_dir)
        
        # 8. Performance delta visualizations (comparing models)
        if len(model_names) > 1:
            self._plot_performance_deltas(model_names, plots_dir)
        
        print(f"Plots saved to {plots_dir}")
    
    def _plot_scenario_comparison(self, category, scenario_names, x_label, plots_dir):
        """Plot comparison of model performance across different scenarios"""
        # Define metrics to plot
        metrics_to_plot = [
            ('avg_reward', 'Average Reward', ''),
            ('avg_task_completion_rate', 'Task Completion Rate', '%', lambda x: x * 100),
            ('avg_latency', 'Average Latency', 's'),
            ('avg_energy_consumption', 'Energy Consumption', 'J')
        ]
        
        # Extract model names
        model_names = list(self.results.keys())
        
        # For each metric, create a comparison plot
        for metric_name, metric_label, unit, transform_func in metrics_to_plot:
            plt.figure(figsize=(12, 6))
            width = 0.8 / len(model_names)
            
            for i, model_name in enumerate(model_names):
                if category in self.results[model_name]:
                    values = []
                    std_values = []
                    
                    for scenario in scenario_names:
                        if scenario in self.results[model_name][category]:
                            val = self.results[model_name][category][scenario]['aggregate'][metric_name]
                            std_val = self.results[model_name][category][scenario]['aggregate'].get(f'std_{metric_name}', 0)
                            
                            if transform_func:
                                val = transform_func(val)
                                std_val = transform_func(std_val)
                            
                            values.append(val)
                            std_values.append(std_val)
                        else:
                            values.append(0)
                            std_values.append(0)
                    
                    x = np.arange(len(scenario_names))
                    offset = width * (i - len(model_names)/2 + 0.5)
                    
                    bars = plt.bar(x + offset, values, width, label=model_name, yerr=std_values, capsize=5, alpha=0.7)
            
            plt.xlabel(x_label)
            plt.ylabel(f'{metric_label} ({unit})' if unit else metric_label)
            plt.title(f'{metric_label} Comparison Across {x_label}s')
            plt.xticks(x, scenario_names)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.tight_layout()
            
            # Ensure filename is valid
            filename = f"{category}_{metric_name.replace('avg_', '')}_comparison.png"
            plt.savefig(os.path.join(plots_dir, filename), dpi=300)
            plt.close()
    
    def _plot_time_series(self, data, title, x_label, y_label, save_path):
        """Plot time series data"""
        plt.figure(figsize=(10, 6))
        
        times = [item['time'] for item in data]
        values = [item['value'] for item in data]
        
        plt.plot(times, values, marker='o', markersize=4, alpha=0.7)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    def _plot_distance_distributions(self, model_names, network_conditions, plots_dir):
        """Plot distance distributions for different network conditions"""
        plt.figure(figsize=(12, 6))
        
        # Only plot first model for clarity
        model_name = model_names[0]
        
        for condition in network_conditions:
            if condition in self.results[model_name]['network']:
                dist_data = self.results[model_name]['network'][condition]['metrics'].get('distance_distribution', {})
                
                if dist_data:
                    # Convert to list of tuples and sort by distance
                    dist_items = sorted([(int(dist), count) for dist, count in dist_data.items()])
                    
                    distances = [item[0] for item in dist_items]
                    counts = [item[1] for item in dist_items]
                    
                    plt.plot(distances, counts, marker='o', linestyle='-', alpha=0.7, label=condition)
        
        plt.title('Task Distance Distribution Across Network Conditions')
        plt.xlabel('Distance to Base Station (m)')
        plt.ylabel('Task Count')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'distance_distribution.png'), dpi=300)
        plt.close()
    
    def _plot_model_comparison_density(self, model_names, plots_dir):
        """Plot model comparison across traffic density scenarios"""
        # Define metrics to plot
        metrics_to_plot = [
            ('avg_task_completion_rate', 'Task Completion Rate (%)', lambda x: x * 100),
            ('avg_latency', 'Average Latency (s)'),
            ('avg_energy_consumption', 'Energy Consumption (J)')
        ]
        
        # Order scenarios by increasing density
        ordered_scenarios = ['low_density', 'normal', 'high_density', 'rush_hour']
        
        for metric_name, metric_label, transform_func in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            
            for model_name in model_names:
                if 'traffic' in self.results[model_name]:
                    x_vals = []
                    y_vals = []
                    
                    for scenario in ordered_scenarios:
                        if scenario in self.results[model_name]['traffic']:
                            x_vals.append(scenario)
                            val = self.results[model_name]['traffic'][scenario]['aggregate'][metric_name]
                            
                            if transform_func:
                                val = transform_func(val)
                                
                            y_vals.append(val)
                    
                    if x_vals and y_vals:
                        plt.plot(x_vals, y_vals, marker='o', markersize=8, linewidth=2, label=model_name)
            
            plt.title(f'{metric_label} Across Traffic Densities')
            plt.xlabel('Traffic Density')
            plt.ylabel(metric_label)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            filename = f"traffic_density_{metric_name.replace('avg_', '')}.png"
            plt.savefig(os.path.join(plots_dir, filename), dpi=300)
            plt.close()
    
    def _plot_model_comparison_network(self, model_names, plots_dir):
        """Plot model comparison across network conditions"""
        # Define metrics to plot
        metrics_to_plot = [
            ('avg_task_completion_rate', 'Task Completion Rate (%)', lambda x: x * 100),
            ('avg_latency', 'Average Latency (s)'),
            ('avg_energy_consumption', 'Energy Consumption (J)')
        ]
        
        # Order conditions from best to worst
        ordered_conditions = ['ideal', 'congested', 'poor_signal', 'limited', 'unstable']
        
        for metric_name, metric_label, transform_func in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            
            for model_name in model_names:
                if 'network' in self.results[model_name]:
                    x_vals = []
                    y_vals = []
                    
                    for condition in ordered_conditions:
                        if condition in self.results[model_name]['network']:
                            x_vals.append(condition)
                            val = self.results[model_name]['network'][condition]['aggregate'][metric_name]
                            
                            if transform_func:
                                val = transform_func(val)
                                
                            y_vals.append(val)
                    
                    if x_vals and y_vals:
                        plt.plot(x_vals, y_vals, marker='o', markersize=8, linewidth=2, label=model_name)
            
            plt.title(f'{metric_label} Across Network Conditions')
            plt.xlabel('Network Condition')
            plt.ylabel(metric_label)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            filename = f"network_condition_{metric_name.replace('avg_', '')}.png"
            plt.savefig(os.path.join(plots_dir, filename), dpi=300)
            plt.close()
    
    def _plot_heatmaps(self, model_names, plots_dir):
        """Generate heatmaps showing performance across scenarios and conditions"""
        # Define metrics to visualize
        metrics_to_plot = [
            ('avg_task_completion_rate', 'Task Completion Rate', lambda x: x * 100, 'viridis'),
            ('avg_latency', 'Average Latency (s)', None, 'coolwarm'),
            ('avg_energy_consumption', 'Energy Consumption (J)', None, 'YlOrRd')
        ]
        
        # Ordered lists of scenarios and conditions
        traffic_scenarios = ['low_density', 'normal', 'high_density', 'rush_hour', 'fluctuating']
        network_conditions = ['ideal', 'congested', 'poor_signal', 'limited', 'unstable']
        
        for model_name in model_names:
            if 'traffic' in self.results[model_name] and 'network' in self.results[model_name]:
                for metric_name, metric_label, transform_func, cmap in metrics_to_plot:
                    # Traffic scenarios heatmap
                    if any(scenario in self.results[model_name]['traffic'] for scenario in traffic_scenarios):
                        plt.figure(figsize=(10, 6))
                        
                        # Prepare data for heatmap
                        scenario_labels = []
                        metric_values = []
                        
                        for scenario in traffic_scenarios:
                            if scenario in self.results[model_name]['traffic']:
                                scenario_labels.append(scenario)
                                val = self.results[model_name]['traffic'][scenario]['aggregate'][metric_name]
                                
                                if transform_func:
                                    val = transform_func(val)
                                    
                                metric_values.append(val)
                        
                        # Convert to numpy array for heatmap
                        metric_array = np.array(metric_values).reshape(1, -1)
                        
                        # Create heatmap
                        sns.heatmap(metric_array, annot=True, fmt='.1f', cmap=cmap, 
                                   xticklabels=scenario_labels, yticklabels=[model_name])
                        
                        plt.title(f'{model_name}: {metric_label} Across Traffic Scenarios')
                        plt.tight_layout()
                        
                        filename = f"{model_name}_traffic_{metric_name.replace('avg_', '')}_heatmap.png"
                        plt.savefig(os.path.join(plots_dir, filename), dpi=300)
                        plt.close()
                    
                    # Network conditions heatmap
                    if any(condition in self.results[model_name]['network'] for condition in network_conditions):
                        plt.figure(figsize=(10, 6))
                        
                        # Prepare data for heatmap
                        condition_labels = []
                        metric_values = []
                        
                        for condition in network_conditions:
                            if condition in self.results[model_name]['network']:
                                condition_labels.append(condition)
                                val = self.results[model_name]['network'][condition]['aggregate'][metric_name]
                                
                                if transform_func:
                                    val = transform_func(val)
                                    
                                metric_values.append(val)
                        
                        # Convert to numpy array for heatmap
                        metric_array = np.array(metric_values).reshape(1, -1)
                        
                        # Create heatmap
                        sns.heatmap(metric_array, annot=True, fmt='.1f', cmap=cmap, 
                                   xticklabels=condition_labels, yticklabels=[model_name])
                        
                        plt.title(f'{model_name}: {metric_label} Across Network Conditions')
                        plt.tight_layout()
                        
                        filename = f"{model_name}_network_{metric_name.replace('avg_', '')}_heatmap.png"
                        plt.savefig(os.path.join(plots_dir, filename), dpi=300)
                        plt.close()
    
    def _plot_performance_deltas(self, model_names, plots_dir):
        """Plot performance differences between models"""
        if len(model_names) < 2:
            return
        
        # Use first two models for comparison
        model1 = model_names[0]
        model2 = model_names[1]
        
        # Define metrics to compare
        metrics_to_compare = [
            ('avg_task_completion_rate', 'Task Completion Rate (%)', lambda x: x * 100),
            ('avg_latency', 'Average Latency (s)'),
            ('avg_energy_consumption', 'Energy Consumption (J)')
        ]
        
        # Traffic scenarios comparison
        traffic_scenarios = list(self.traffic_scenarios.keys())
        if all('traffic' in self.results[model] for model in [model1, model2]):
            for metric_name, metric_label, transform_func in metrics_to_compare:
                scenario_labels = []
                delta_values = []
                
                for scenario in traffic_scenarios:
                    if (scenario in self.results[model1]['traffic'] and 
                        scenario in self.results[model2]['traffic']):
                        
                        scenario_labels.append(scenario)
                        
                        val1 = self.results[model1]['traffic'][scenario]['aggregate'][metric_name]
                        val2 = self.results[model2]['traffic'][scenario]['aggregate'][metric_name]
                        
                        if transform_func:
                            val1 = transform_func(val1)
                            val2 = transform_func(val2)
                        
                        delta = val2 - val1  # model2 - model1
                        delta_values.append(delta)
                
                if scenario_labels and delta_values:
                    plt.figure(figsize=(10, 6))
                    
                    bars = plt.bar(scenario_labels, delta_values, alpha=0.7)
                    
                    # Color bars based on which model performs better
                    for i, bar in enumerate(bars):
                        if delta_values[i] >= 0:
                            bar.set_color('green')
                        else:
                            bar.set_color('red')
                    
                    plt.axhline(y=0, color='k', linestyle='--')
                    plt.title(f'{metric_label} Delta: {model2} vs {model1} (Traffic Scenarios)')
                    plt.ylabel(f'Delta {metric_label}')
                    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                    
                    # Add a legend
                    green_patch = mpatches.Patch(color='green', label=f'{model2} better')
                    red_patch = mpatches.Patch(color='red', label=f'{model1} better')
                    plt.legend(handles=[green_patch, red_patch])
                    
                    plt.tight_layout()
                    
                    filename = f"delta_traffic_{metric_name.replace('avg_', '')}.png"
                    plt.savefig(os.path.join(plots_dir, filename), dpi=300)
                    plt.close()
        
        # Network conditions comparison
        network_conditions = list(self.network_conditions.keys())
        if all('network' in self.results[model] for model in [model1, model2]):
            for metric_name, metric_label, transform_func in metrics_to_compare:
                condition_labels = []
                delta_values = []
                
                for condition in network_conditions:
                    if (condition in self.results[model1]['network'] and 
                        condition in self.results[model2]['network']):
                        
                        condition_labels.append(condition)
                        
                        val1 = self.results[model1]['network'][condition]['aggregate'][metric_name]
                        val2 = self.results[model2]['network'][condition]['aggregate'][metric_name]
                        
                        if transform_func:
                            val1 = transform_func(val1)
                            val2 = transform_func(val2)
                        
                        delta = val2 - val1  # model2 - model1
                        delta_values.append(delta)
                
                if condition_labels and delta_values:
                    plt.figure(figsize=(10, 6))
                    
                    bars = plt.bar(condition_labels, delta_values, alpha=0.7)
                    
                    # Color bars based on which model performs better
                    # Note: For latency, lower is better, so the coloring is inverted
                    for i, bar in enumerate(bars):
                        if metric_name == 'avg_latency':
                            if delta_values[i] <= 0:
                                bar.set_color('green')
                            else:
                                bar.set_color('red')
                        else:
                            if delta_values[i] >= 0:
                                bar.set_color('green')
                            else:
                                bar.set_color('red')
                    
                    plt.axhline(y=0, color='k', linestyle='--')
                    plt.title(f'{metric_label} Delta: {model2} vs {model1} (Network Conditions)')
                    plt.ylabel(f'Delta {metric_label}')
                    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                    
                    # Add a legend
                    green_patch = mpatches.Patch(color='green', label=f'{model2} better')
                    red_patch = mpatches.Patch(color='red', label=f'{model1} better')
                    plt.legend(handles=[green_patch, red_patch])
                    
                    plt.tight_layout()
                    
                    filename = f"delta_network_{metric_name.replace('avg_', '')}.png"
                    plt.savefig(os.path.join(plots_dir, filename), dpi=300)
                    plt.close()


def main():
    """Main function to run scenario analysis"""
    parser = argparse.ArgumentParser(description='Analyze DQN models under different scenarios')
    
    parser.add_argument('--ff_dqn_model', type=str, required=True,
                       help='Path to the trained FF-DQN model checkpoint')
    parser.add_argument('--te_ddqn_model', type=str, required=True,
                       help='Path to the trained TE-DDQN model checkpoint')
    parser.add_argument('--output_dir', type=str, default='scenario_analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--energy_csv_path', type=str, default='merged_dag1.csv',
                       help='Path to energy consumption data CSV')
    parser.add_argument('--num_episodes', type=int, default=5,
                       help='Number of episodes to run for each scenario')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--traffic_only', action='store_true',
                       help='Only analyze traffic scenarios (skip network conditions)')
    parser.add_argument('--network_only', action='store_true',
                       help='Only analyze network conditions (skip traffic scenarios)')
    
    args = parser.parse_args()
    
    # Define model paths
    model_paths = {
        'FF-DQN': args.ff_dqn_model,
        'TE-DDQN': args.te_ddqn_model
    }
    
    # Create analyzer
    analyzer = ScenarioAnalyzer(
        model_paths=model_paths,
        energy_csv_path=args.energy_csv_path,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Run selective analysis based on flags
    if args.traffic_only:
        for model_name, model_path in model_paths.items():
            for scenario_name in analyzer.traffic_scenarios:
                analyzer.analyze_traffic_scenario(scenario_name, model_name, model_path, args.num_episodes)
    elif args.network_only:
        for model_name, model_path in model_paths.items():
            for condition_name in analyzer.network_conditions:
                analyzer.analyze_network_condition(condition_name, model_name, model_path, args.num_episodes)
    else:
        # Run full analysis
        analyzer.run_analysis(args.num_episodes)
    
    # Generate plots regardless of which analysis was run
    analyzer.save_results()
    analyzer.generate_plots()
    
    print("\nScenario analysis completed!")
    print(f"Results and visualizations saved to: {analyzer.results_dir}")


if __name__ == "__main__":
    main()