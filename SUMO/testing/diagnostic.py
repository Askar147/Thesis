#!/usr/bin/env python3
"""
Diagnostic script to locate model files from TE-DQN training
"""

import os
import time
from datetime import datetime

def find_model_files(base_dir="te_ddqn_curriculum", search_for=".pt"):
    """Find all model files in the directory structure"""
    print(f"Searching for model files ({search_for}) in {base_dir}")
    
    all_models = []
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist!")
        return all_models
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if search_for in file:
                full_path = os.path.join(root, file)
                file_size = os.path.getsize(full_path)
                file_time = os.path.getmtime(full_path)
                time_str = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
                
                all_models.append({
                    'path': full_path,
                    'size': file_size,
                    'time': time_str
                })
    
    # Sort by modification time (newest first)
    all_models.sort(key=lambda x: x['path'])
    
    # Print results
    if all_models:
        print(f"Found {len(all_models)} model files:")
        for i, model in enumerate(all_models):
            print(f"{i+1}. {model['path']} ({model['size']/1024:.2f} KB, modified: {model['time']})")
    else:
        print("No model files found!")
    
    return all_models

def check_model_directories():
    """Check if model directories exist and report their contents"""
    directories = [
        "te_ddqn_curriculum/phase1",
        "te_ddqn_curriculum/phase2",
        "te_ddqn_curriculum/phase3",
        "models"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"Directory exists: {directory}")
            dir_contents = os.listdir(directory)
            if dir_contents:
                print(f"  Contents ({len(dir_contents)} items): {dir_contents[:5]}...")
                if len(dir_contents) > 5:
                    print(f"  ...and {len(dir_contents)-5} more items")
            else:
                print("  Directory is empty")
        else:
            print(f"Directory does not exist: {directory}")

if __name__ == "__main__":
    print("=" * 80)
    print("TE-DQN Training Diagnostic")
    print("=" * 80)
    print()
    
    # Check directories
    print("Checking directories...")
    check_model_directories()
    print()
    
    # Find model files
    print("Searching for model files...")
    find_model_files()
    print()
    
    # Look in stress test results
    print("Checking stress test results...")
    find_model_files("models/te_ddqn_test_results")
    print()
    
    print("Diagnostic complete")