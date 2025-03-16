#!/usr/bin/env python3
"""
Main script to run the Transformer-based agent for VEC task offloading
with energy-aware environment
"""

import os
import argparse
import json
from datetime import datetime

from vec_environment_2 import VECEnvironment
from vec_transformer_agent_2 import (
    VECTransformerAgent, 
    get_state_size, 
    flatten_observation,
    train_vec_transformer,
    evaluate_transformer_agent,
    save_transformer_training_results,
    compare_models
)

def main():
    """Main function to run VEC transformer experiments with energy-aware environment"""
    parser = argparse.ArgumentParser(description='Run VEC transformer experiments with energy-aware environment')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'compare'], default='train',
                       help='Mode to run: train, evaluate, or compare')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (JSON)')
    parser.add_argument('--output_dir', type=str, default='transformer_results',
                       help='Directory to save results')
    parser.add_argument('--energy_csv_path', type=str, default=None,
                       help='Path to energy consumption data CSV (optional)')
    parser.add_argument('--energy_weight', type=float, default=0.5,
                       help='Weight for energy consumption in reward (0.0-1.0)')
    
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
        # Default configuration with energy parameters
        config = {
            "environment": {
                "sumo_config": "astana.sumocfg",
                "simulation_duration": 300,
                "time_step": 1,
                "queue_process_interval": 5,
                "max_queue_length": 50,
                "history_length": 10,
                "energy_csv_path": args.energy_csv_path,
                "energy_weight": args.energy_weight,
                "seed": 42
            },
            "agent": {
                "seq_length": 16,
                "d_model": 128,
                "nhead": 4,
                "num_layers": 3,
                "dropout": 0.1,
                "batch_size": 64,
                "gamma": 0.99,
                "tau": 0.01,
                "min_replay_size": 1000,
                "epsilon_min": 0.05,
                "epsilon_decay_steps": 2000,
                "lr": 0.0003,
                "weight_decay": 1e-5,
                "load_model": None
            },
            "training": {
                "num_episodes": 1000,
                "max_steps": 300,
                "eval_frequency": 50
            },
            "evaluation": {
                "episodes": 10,
                "seed": 123
            },
            "compare": {
                "models": []
            }
        }
        print("Using default configuration with energy parameters")
    
    # Ensure energy parameters are set if provided via command line
    if args.energy_csv_path:
        config['environment']['energy_csv_path'] = args.energy_csv_path
    
    if args.energy_weight:
        config['environment']['energy_weight'] = args.energy_weight
    
    # Save configuration
    with open(os.path.join(run_dir, 'transformer_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Prepare environment config for training/evaluation
    env_config = {
        'sumo_config': config['environment']['sumo_config'],
        'simulation_duration': config['environment']['simulation_duration'],
        'time_step': config['environment']['time_step'],
        'queue_process_interval': config['environment']['queue_process_interval'],
        'max_queue_length': config['environment']['max_queue_length'],
        'history_length': config['environment']['history_length'],
        'energy_csv_path': config['environment'].get('energy_csv_path'),
        'energy_weight': config['environment'].get('energy_weight', 0.5),
        'seed': config['environment']['seed'],
        'num_episodes': config['training']['num_episodes'],
        'max_steps': config['training']['max_steps']
    }
    
    # Run in the specified mode
    if args.mode == 'train':
        print("Starting transformer training with energy-aware environment...")
        
        # Set agent parameters
        agent_params = config['agent']
        
        agent, metrics = train_vec_transformer(env_config)
        save_transformer_training_results(metrics, env_config, save_dir=run_dir)
        
        print(f"Training completed. Results saved to {run_dir}")
        
    elif args.mode == 'evaluate':
        if not config['agent'].get('load_model'):
            print("Error: No model specified for evaluation")
            return
            
        print(f"Evaluating transformer model: {config['agent']['load_model']}")
        
        # Create environment
        eval_env = VECEnvironment(
            sumo_config=config['environment']['sumo_config'],
            simulation_duration=config['environment']['simulation_duration'],
            time_step=config['environment']['time_step'],
            queue_process_interval=config['environment']['queue_process_interval'],
            max_queue_length=config['environment']['max_queue_length'],
            history_length=config['environment']['history_length'],
            energy_csv_path=config['environment'].get('energy_csv_path'),
            energy_weight=config['environment'].get('energy_weight', 0.5),
            seed=config['evaluation']['seed']  # Different seed for evaluation
        )
        
        # Calculate state size
        state_size = get_state_size(eval_env)
        action_size = eval_env.action_space.n
        
        # Create agent
        agent = VECTransformerAgent(state_size, action_size)
        
        # Load pretrained model
        agent.load_model(config['agent']['load_model'])
        
        # Evaluate
        eval_results = evaluate_transformer_agent(
            agent, 
            eval_env, 
            num_episodes=config['evaluation']['episodes']
        )
        
        # Save evaluation results
        eval_results_path = os.path.join(run_dir, "transformer_evaluation_results.json")
        with open(eval_results_path, 'w') as f:
            json.dump(eval_results, f, indent=4)
        
        # Print evaluation results
        print("\nEvaluation Results:")
        for metric, value in eval_results.items():
            print(f"{metric}: {value:.4f}")
        
        print(f"Evaluation results saved to {eval_results_path}")
        
    elif args.mode == 'compare':
        models = config['compare'].get('models', [])
        if not models or len(models) < 2:
            print("Error: Need at least two models for comparison")
            return
        
        dqn_model = next((m for m in models if "DQN" in m['name']), None)
        transformer_model = next((m for m in models if "Transformer" in m['name']), None)
        
        if not dqn_model or not transformer_model:
            print("Error: Need both DQN and Transformer models for comparison")
            return
            
        dqn_dir = os.path.dirname(dqn_model['path'])
        transformer_dir = os.path.dirname(transformer_model['path'])
        
        print(f"Comparing DQN model from {dqn_dir} with Transformer model from {transformer_dir}")
        
        compare_models(dqn_dir, transformer_dir, save_dir=run_dir)
        
        print(f"Comparison results saved to {run_dir}")
    
    print("Done!")

if __name__ == "__main__":
    main()