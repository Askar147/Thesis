#!/usr/bin/env python3
"""
Modified train_dqn_main.py that handles SUMO connection issues
"""

import os
import argparse
import json
from datetime import datetime
import time
import sys

def main():
    """Main function to run FF-DQN training with enhanced environment"""
    parser = argparse.ArgumentParser(description='Train FF-DQN agent with enhanced VEC environment')
    parser.add_argument('--sumo_config', type=str, default='astana.sumocfg', 
                        help='Path to SUMO configuration file')
    parser.add_argument('--energy_csv', type=str, required=True,
                        help='Path to energy consumption data CSV (required)')
    parser.add_argument('--energy_weight', type=float, default=0.7,
                        help='Weight for energy consumption in reward (0.0-1.0)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=5000,
                        help='Maximum steps per episode')
    parser.add_argument('--duration', type=int, default=300,
                        help='Simulation duration in seconds')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--min_tasks', type=int, default=10,
                        help='Minimum tasks per step')
    parser.add_argument('--max_tasks', type=int, default=20,
                        help='Maximum tasks per step')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load existing model (optional)')
    
    args = parser.parse_args()
    
    # Make sure any existing TRACI connections are closed
    try:
        import traci
        if traci.isConnected():
            traci.close()
            print("Closed existing TRACI connection")
            time.sleep(5)  # Give time for cleanup
    except Exception as e:
        print(f"Error closing TRACI: {e}")
    
    # Try to forcefully terminate any SUMO processes (Windows)
    try:
        import subprocess
        subprocess.run(['taskkill', '/F', '/IM', 'sumo.exe'], 
                      stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        subprocess.run(['taskkill', '/F', '/IM', 'sumo-gui.exe'], 
                      stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        print("Forcefully terminated any existing SUMO processes")
        time.sleep(5)  # Give time for cleanup
    except Exception as e:
        print(f"Error terminating SUMO processes: {e}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"ff_dqn_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Environment configuration
    env_config = {
        'sumo_config': args.sumo_config,
        'simulation_duration': args.duration,
        'time_step': 1,
        'queue_process_interval': 5,
        'max_queue_length': 50,
        'history_length': 10,
        'energy_csv_path': args.energy_csv,
        'energy_weight': args.energy_weight,
        'latency_model_params': {
            'frequency_band': 2.4,
            'bandwidth': 20,
            'noise_floor': -95
        },
        'min_tasks_per_step': args.min_tasks,
        'max_tasks_per_step': args.max_tasks,
        'task_generation_probability': 1,
        'seed': args.seed
    }
    
    # Agent configuration
    agent_config = {
        'num_episodes': args.episodes,
        'max_steps': args.max_steps,
        'target_update_frequency': 10,
        'load_model': args.load_model
    }
    
    # Save configuration
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({'environment': env_config, 'agent': agent_config}, f, indent=4)
    
    print(f"Starting FF-DQN training with enhanced environment...")
    print(f"Using energy data from: {args.energy_csv}")
    print(f"Energy weight: {args.energy_weight}")
    print(f"Episodes: {args.episodes}")
    print(f"Output directory: {run_dir}")
    
    # Import training module here to ensure clean environment
    from ff_dqn_train import train_dqn
    
    # Run training
    try:
        agent, metrics, output_dir = train_dqn(env_config, agent_config, log_dir=run_dir)
        
        print(f"Training completed!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
    # Ensure TRACI connections are closed when we're done
    try:
        import traci
        if traci.isConnected():
            traci.close()
    except:
        pass

if __name__ == "__main__":
    main()