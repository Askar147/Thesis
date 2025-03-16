#!/usr/bin/env python3
"""
Main script to run VEC DQN experiments
"""

import os
import argparse
from datetime import datetime
import json

from vec_environment import VECEnvironment
from vec_dqn_agent import DQNAgent
from vec_dqn_train import train_dqn
from vec_dqn_evaluate import evaluate_dqn, compare_models


def main():
    """Main function to run VEC DQN experiments"""
    parser = argparse.ArgumentParser(description='Run VEC DQN experiments')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'compare'], default='train',
                       help='Mode to run: train, evaluate, or compare')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (JSON)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{args.mode}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Load configuration from file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.config}")
    else:
        # Default configuration
        config = {
            'environment': {
                'sumo_config': 'astana.sumocfg',
                'simulation_duration': 300,
                'time_step': 1,
                'queue_process_interval': 5,
                'max_queue_length': 50,
                'history_length': 10,
                'seed': 42
            },
            'agent': {
                'num_episodes': 1000,
                'max_steps': 500,
                'target_update_frequency': 10,
                'load_model': None
            },
            'evaluation': {
                'episodes': 10,
                'seed': 123,
                'render': False
            },
            'compare': {
                'models': []
            }
        }
        print("Using default configuration")
    
    # Save configuration
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Run in the specified mode
    if args.mode == 'train':
        print("Starting training...")
        agent, metrics, train_dir = train_dqn(
            config['environment'],
            config['agent'],
            log_dir=run_dir
        )
        print(f"Training completed. Results saved to {train_dir}")
        
    elif args.mode == 'evaluate':
        if not config['agent'].get('load_model'):
            print("Error: No model specified for evaluation")
            return
        
        print(f"Evaluating model: {config['agent']['load_model']}")
        results = evaluate_dqn(
            config['agent']['load_model'],
            config['environment'],
            config['evaluation'].get('episodes', 10),
            config['evaluation'].get('render', False)
        )
        
        if results:
            # Save evaluation results
            results_path = os.path.join(run_dir, 'evaluation_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Evaluation results saved to {results_path}")
        
    elif args.mode == 'compare':
        models = config['compare'].get('models', [])
        if not models:
            print("Error: No models specified for comparison")
            return
        
        print(f"Comparing {len(models)} models...")
        compare_models(
            models,
            config['environment'],
            config['evaluation'].get('episodes', 10),
            save_dir=run_dir
        )
        print(f"Comparison results saved to {run_dir}")
    
    print("Done!")


if __name__ == "__main__":
    main()