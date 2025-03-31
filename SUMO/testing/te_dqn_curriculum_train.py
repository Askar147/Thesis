"""
TE-DDQN curriculum training script for Windows
"""

import os
import subprocess
import time
import json
from datetime import datetime

# Set main variables
BASE_DIR = "te_ddqn_curriculum"
ENERGY_CSV = "merged_dag1.csv"
TOTAL_EPISODES = 1000

# Create base directory
os.makedirs(BASE_DIR, exist_ok=True)
print(f"Starting TE-DDQN curriculum training in {BASE_DIR}")

# Phase 1: Basic training with low load
print("Phase 1: Low load training (200 episodes)")
phase1_dir = os.path.join(BASE_DIR, "phase1")
os.makedirs(phase1_dir, exist_ok=True)

phase1_cmd = [
    "python", "train_transformer_main.py",
    "--energy_csv", ENERGY_CSV,
    "--energy_weight", "0.7",
    "--episodes", "200",
    "--max_steps", "1000",
    "--duration", "300",
    "--output_dir", phase1_dir,
    "--min_tasks", "10",
    "--max_tasks", "15",
    "--seq_length", "16",
    "--d_model", "128",
    "--nheads", "4"
]

# Run phase 1
subprocess.run(phase1_cmd)

# Check if best model was generated
phase1_model_path = os.path.join(phase1_dir, "best_model.pt")
if not os.path.isfile(phase1_model_path):
    print("Error: Phase 1 did not generate a best model. Check logs.")
    exit(1)

# Phase 2: Medium load training
print("Phase 2: Medium load training (300 episodes)")
phase2_dir = os.path.join(BASE_DIR, "phase2")
os.makedirs(phase2_dir, exist_ok=True)

phase2_cmd = [
    "python", "train_transformer_main.py",
    "--energy_csv", ENERGY_CSV,
    "--energy_weight", "0.6",
    "--episodes", "300",
    "--max_steps", "1000",
    "--duration", "300",
    "--output_dir", phase2_dir,
    "--min_tasks", "20",
    "--max_tasks", "30",
    "--seq_length", "16",
    "--d_model", "128",
    "--nheads", "4",
    "--load_model", phase1_model_path
]

# Run phase 2
subprocess.run(phase2_cmd)

# Check if best model was generated
phase2_model_path = os.path.join(phase2_dir, "best_model.pt")
if not os.path.isfile(phase2_model_path):
    print("Error: Phase 2 did not generate a best model. Using Phase 1 model.")
    # Copy phase 1 model if phase 2 model is missing
    import shutil
    shutil.copy(phase1_model_path, phase2_model_path)

# Phase 3a: High load training with reduced sequence length
print("Phase 3a: High load training with reduced sequence length (250 episodes)")
phase3a_dir = os.path.join(BASE_DIR, "phase3a")
os.makedirs(phase3a_dir, exist_ok=True)

phase3a_cmd = [
    "python", "train_transformer_main.py",
    "--energy_csv", ENERGY_CSV,
    "--energy_weight", "0.5",
    "--episodes", "250",
    "--max_steps", "1000", 
    "--duration", "300",
    "--output_dir", phase3a_dir,
    "--min_tasks", "40",
    "--max_tasks", "50",
    "--seq_length", "8",  # Shorter sequence length for faster adaptation
    "--d_model", "128",
    "--nheads", "4",
    "--load_model", phase2_model_path
]

# Run phase 3a
subprocess.run(phase3a_cmd)

# Phase 3b: Continue high load training with original sequence length
print("Phase 3b: Continue high load training with original sequence length (250 episodes)")
phase3b_dir = os.path.join(BASE_DIR, "phase3b")
os.makedirs(phase3b_dir, exist_ok=True)

# Check if phase 3a model exists
phase3a_model_path = os.path.join(phase3a_dir, "best_model.pt")
prev_model_path = phase3a_model_path if os.path.isfile(phase3a_model_path) else phase2_model_path

phase3b_cmd = [
    "python", "train_transformer_main.py",
    "--energy_csv", ENERGY_CSV,
    "--energy_weight", "0.5",
    "--episodes", "250",
    "--max_steps", "1000",
    "--duration", "300",
    "--output_dir", phase3b_dir,
    "--min_tasks", "40",
    "--max_tasks", "50",
    "--seq_length", "16",
    "--d_model", "128",
    "--nheads", "4",
    "--load_model", prev_model_path
]

# Run phase 3b
subprocess.run(phase3b_cmd)

# Determine best final model
phase3b_model_path = os.path.join(phase3b_dir, "best_model.pt")
if os.path.isfile(phase3b_model_path):
    final_model_path = phase3b_model_path
elif os.path.isfile(phase3a_model_path):
    final_model_path = phase3a_model_path
else:
    final_model_path = phase2_model_path
    print("Warning: Phase 3 did not generate a best model. Using Phase 2 model.")

# Copy final model to root directory
os.makedirs("models", exist_ok=True)
final_dest_path = os.path.join("models", "te_ddqn_curriculum_final.pt")

import shutil
shutil.copy(final_model_path, final_dest_path)

print("TE-DDQN curriculum training complete")
print(f"Final model saved to {final_dest_path}")

# Test the final model with a high-load scenario
print("Testing final model with high-load scenario")
test_cmd = [
    "python", "high-load-scenario.py",
    "--ff_model", "best_model/best_model_ff.pth",
    "--te_model", final_dest_path,
    "--intensity", "high",
    "--episodes", "2",
    "--output", os.path.join("models", "te_ddqn_test_results")
]

subprocess.run(test_cmd)

print("TE-DDQN curriculum training and testing complete")