"""
FF-DQN curriculum training script for Windows
"""

import os
import subprocess
import time
import json
from datetime import datetime

# Set main variables
BASE_DIR = "ff_dqn_curriculum"
ENERGY_CSV = "merged_dag1.csv"
TOTAL_EPISODES = 1000

# Create base directory
os.makedirs(BASE_DIR, exist_ok=True)
print(f"Starting FF-DQN curriculum training in {BASE_DIR}")

# Phase 1: Basic training with low load
print("Phase 1: Low load training (200 episodes)")
phase1_dir = os.path.join(BASE_DIR, "phase1")
os.makedirs(phase1_dir, exist_ok=True)

phase1_cmd = [
    "python", "train_dqn_main.py",
    "--energy_csv", ENERGY_CSV,
    "--energy_weight", "0.7",
    "--episodes", "200",
    "--max_steps", "1000",
    "--duration", "300",
    "--output_dir", phase1_dir,
    "--min_tasks", "10",
    "--max_tasks", "15",
]

# Run phase 1
subprocess.run(phase1_cmd)

# Check if best model was generated
phase1_model_path = os.path.join(phase1_dir, "best_model.pth")
if not os.path.isfile(phase1_model_path):
    print("Error: Phase 1 did not generate a best model. Check logs.")
    exit(1)

# Phase 2: Medium load training
print("Phase 2: Medium load training (300 episodes)")
phase2_dir = os.path.join(BASE_DIR, "phase2")
os.makedirs(phase2_dir, exist_ok=True)

phase2_cmd = [
    "python", "train_dqn_main.py",
    "--energy_csv", ENERGY_CSV,
    "--energy_weight", "0.6",
    "--episodes", "300",
    "--max_steps", "1000",
    "--duration", "300",
    "--output_dir", phase2_dir,
    "--min_tasks", "20",
    "--max_tasks", "30",
    "--load_model", phase1_model_path
]

# Run phase 2
subprocess.run(phase2_cmd)

# Check if best model was generated
phase2_model_path = os.path.join(phase2_dir, "best_model.pth")
if not os.path.isfile(phase2_model_path):
    print("Error: Phase 2 did not generate a best model. Using Phase 1 model.")
    # Copy phase 1 model if phase 2 model is missing
    import shutil
    shutil.copy(phase1_model_path, phase2_model_path)

# Phase 3: High load training
print("Phase 3: High load training (500 episodes)")
phase3_dir = os.path.join(BASE_DIR, "phase3")
os.makedirs(phase3_dir, exist_ok=True)

phase3_cmd = [
    "python", "train_dqn_main.py",
    "--energy_csv", ENERGY_CSV,
    "--energy_weight", "0.5",
    "--episodes", "500",
    "--max_steps", "1000",
    "--duration", "300",
    "--output_dir", phase3_dir,
    "--min_tasks", "40",
    "--max_tasks", "50",
    "--load_model", phase2_model_path
]

# Run phase 3
subprocess.run(phase3_cmd)

# Copy final model to root directory
os.makedirs("models", exist_ok=True)
phase3_model_path = os.path.join(phase3_dir, "best_model.pth")
final_model_path = os.path.join("models", "ff_dqn_curriculum_final.pth")

import shutil
if os.path.isfile(phase3_model_path):
    shutil.copy(phase3_model_path, final_model_path)
    print(f"FF-DQN curriculum training complete")
    print(f"Final model saved to {final_model_path}")
else:
    print("Warning: Phase 3 did not generate a best model. Using Phase 2 model.")
    shutil.copy(phase2_model_path, final_model_path)
    print(f"FF-DQN curriculum training completed with warning")
    print(f"Final model (from Phase 2) saved to {final_model_path}")

# Test the final model with a high-load scenario
print("Testing final model with high-load scenario")
test_cmd = [
    "python", "high-load-scenario.py",
    "--ff_model", final_model_path,
    "--te_model", "best_model/best_model_te.pt",
    "--intensity", "high",
    "--episodes", "2",
    "--output", os.path.join("models", "ff_dqn_test_results")
]

subprocess.run(test_cmd)

print("FF-DQN curriculum training and testing complete")