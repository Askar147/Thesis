#!/usr/bin/env python3
"""
Run evaluation of FF-DQN and TE-DQN models for VEC task offloading
"""

import os
import argparse
from datetime import datetime
from model_evaluator import ModelEvaluator

def main():
    parser = argparse.ArgumentParser(description='Run VEC model evaluation')
    parser.add_argument('--ff_model', type=str, required=True, 
                        help='Path to the FF-DQN model (best_model.pth or final_model.pth)')
    parser.add_argument('--te_model', type=str, required=True, 
                        help='Path to the TE-DQN model (best_model.pt or final_model.pt)')
    parser.add_argument('--output', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of evaluation episodes per scenario (default: 5)')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Specific scenario to evaluate (optional)')
    parser.add_argument('--max_steps', type=int, default=300,
                        help='Maximum steps per episode (default: 300)')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(args.output, f"eval_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Create evaluator
    evaluator = ModelEvaluator(eval_dir)
    
    # Load models
    print(f"Loading FF-DQN model from: {args.ff_model}")
    print(f"Loading TE-DQN model from: {args.te_model}")
    evaluator.load_models(args.ff_model, args.te_model)
    
    # Run evaluation
    if args.scenario:
        if args.scenario in evaluator.scenarios:
            print(f"Evaluating specific scenario: {args.scenario}")
            results = evaluator.evaluate_scenario(args.scenario, args.episodes, args.max_steps)
        else:
            print(f"Error: Scenario '{args.scenario}' not found")
            print("Available scenarios:")
            for scenario_name, scenario_info in evaluator.scenarios.items():
                print(f"  - {scenario_name}: {scenario_info['name']}")
            return
    else:
        print(f"Running evaluation on all scenarios with {args.episodes} episodes each")
        results = evaluator.run_all_evaluations(args.episodes, args.max_steps)
    
    print(f"Evaluation complete! Results saved to: {eval_dir}")
    print(f"HTML report available at: {os.path.join(eval_dir, 'evaluation_report.html')}")

if __name__ == "__main__":
    main()