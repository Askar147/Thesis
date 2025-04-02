#!/usr/bin/env python3
"""
Enhanced Curriculum Training for TE-DQN
Includes extreme load phase with 500 max tasks and 10000 max steps
"""

import os
import subprocess
import time
import shutil
import sys

# Configuration
ENERGY_CSV = "merged_dag1.csv"
BASE_DIR = "te_dqn_easy_curriculum"
MODELS_DIR = "models"
WAIT_TIME = 60  # Seconds to wait between phases

# Create directories
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Starting Enhanced TE-DQN Curriculum Training in {BASE_DIR}")

# Helper function to kill SUMO processes
def kill_sumo():
    print("Forcefully terminating SUMO processes...")
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'sumo.exe'], 
                      stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        subprocess.run(['taskkill', '/F', '/IM', 'sumo-gui.exe'], 
                      stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        print("SUMO processes terminated. Waiting for cleanup...")
        time.sleep(10)  # Give enough time
    except Exception as e:
        print(f"Error terminating SUMO: {e}")

# Helper to run a phase and look for models
def run_phase(phase_name, phase_dir, cmd):
    print(f"\n{'='*80}\n{phase_name}\n{'='*80}\n")
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    print(f"Command completed with exit code: {result.returncode}")
    
    # Look for models
    models = []
    for root, dirs, files in os.walk(phase_dir):
        for file in files:
            if file.endswith(".pt"):
                model_path = os.path.join(root, file)
                models.append(model_path)
                size_kb = os.path.getsize(model_path) / 1024
                print(f"Found model: {model_path} ({size_kb:.2f} KB)")
    
    if models:
        # Sort by modification time (newest first)
        models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return models[0]  # Return newest model
    return None

# Phase 1: Low load training
phase1_dir = os.path.join(BASE_DIR, "phase1")
os.makedirs(phase1_dir, exist_ok=True)

# Kill any existing SUMO processes before starting
kill_sumo()

# # Run Phase 1
# phase1_cmd = [
#     "python", "fix_train_transformer_main.py",  # Use our fixed version
#     "--energy_csv", ENERGY_CSV,
#     "--energy_weight", "0.7",
#     "--episodes", "200",
#     "--max_steps", "1000",
#     "--duration", "300",
#     "--output_dir", phase1_dir,
#     "--min_tasks", "10",
#     "--max_tasks", "15",
#     "--seq_length", "16",
#     "--d_model", "128",
#     "--nheads", "4",
#     "--skip_evaluation"  # Important: skip evaluation to avoid input prompt
# ]

# phase1_model = run_phase("Phase 1: Low Load Training", phase1_dir, phase1_cmd)

# if not phase1_model:
#     print("Error: Phase 1 did not produce a model. Exiting.")
#     sys.exit(1)

# # Copy Phase 1 model to checkpoint
# checkpoint1 = os.path.join(MODELS_DIR, "te_dqn_phase1.pt")
# shutil.copy(phase1_model, checkpoint1)
# print(f"Phase 1 model saved to: {checkpoint1}")

# # Wait between phases
# print(f"Waiting {WAIT_TIME} seconds between phases...")
# time.sleep(WAIT_TIME)

# # Kill SUMO processes before Phase 2
# kill_sumo()

# Phase 2: Medium load training
phase2_dir = os.path.join(BASE_DIR, "phase2")
os.makedirs(phase2_dir, exist_ok=True)

# Run Phase 2
phase2_cmd = [
    "python", "fix_train_transformer_main.py",  # Use our fixed version
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
    # "--load_model", phase1_model,
    "--skip_evaluation"  # Important: skip evaluation to avoid input prompt
]

phase2_model = run_phase("Phase 2: Medium Load Training", phase2_dir, phase2_cmd)

# if not phase2_model:
#     print("Warning: Phase 2 did not produce a model. Using Phase 1 model.")
#     phase2_model = phase1_model
# else:
#     # Copy Phase 2 model to checkpoint
#     checkpoint2 = os.path.join(MODELS_DIR, "te_dqn_phase2.pt")
#     shutil.copy(phase2_model, checkpoint2)
#     print(f"Phase 2 model saved to: {checkpoint2}")

# Wait between phases
print(f"Waiting {WAIT_TIME} seconds between phases...")
time.sleep(WAIT_TIME)

# Kill SUMO processes before Phase 3
kill_sumo()

# Phase 3: High load training
phase3_dir = os.path.join(BASE_DIR, "phase3")
os.makedirs(phase3_dir, exist_ok=True)

# Run Phase 3
phase3_cmd = [
    "python", "fix_train_transformer_main.py",  # Use our fixed version
    "--energy_csv", ENERGY_CSV,
    "--energy_weight", "0.5",
    "--episodes", "500",
    "--max_steps", "1000",
    "--duration", "300",
    "--output_dir", phase3_dir,
    "--min_tasks", "40",
    "--max_tasks", "50",
    "--seq_length", "16",
    "--d_model", "128",
    "--nheads", "4",
    # "--load_model", phase2_model,
    "--skip_evaluation"  # Important: skip evaluation to avoid input prompt
]

phase3_model = run_phase("Phase 3: High Load Training", phase3_dir, phase3_cmd)

# if not phase3_model:
#     print("Warning: Phase 3 did not produce a model. Using Phase 2 model.")
#     phase3_model = phase2_model

# Copy Phase 3 model to checkpoint if available
if phase3_model:
    checkpoint3 = os.path.join(MODELS_DIR, "te_dqn_phase3.pt")
    shutil.copy(phase3_model, checkpoint3)
    print(f"Phase 3 model saved to: {checkpoint3}")

# Wait between phases
print(f"Waiting {WAIT_TIME} seconds between phases...")
time.sleep(WAIT_TIME)

# Kill SUMO processes before Phase 4
kill_sumo()

# Phase 4: Extreme load training (New phase with 500 max tasks)
phase4_dir = os.path.join(BASE_DIR, "phase4_extreme")
os.makedirs(phase4_dir, exist_ok=True)

# Use the best model from previous phases
best_model = phase3_model if phase3_model else phase2_model if phase2_model else phase1_model

# Run Phase 4 with extreme load
phase4_cmd = [
    "python", "fix_train_transformer_main.py",  # Use our fixed version
    "--energy_csv", ENERGY_CSV,
    "--energy_weight", "0.5",
    "--episodes", "400",
    "--max_steps", "10000",  # Significantly increased max steps
    "--duration", "500",     # Longer simulation duration
    "--output_dir", phase4_dir,
    "--min_tasks", "450",    # Much higher task load
    "--max_tasks", "500",    # Extreme task generation
    "--seq_length", "16",
    "--d_model", "128",
    "--nheads", "4",
    # "--load_model", best_model,
    "--skip_evaluation"      # Skip evaluation to avoid input prompt
]

phase4_model = run_phase("Phase 4: Extreme Load Training (500 max tasks)", phase4_dir, phase4_cmd)

# Determine final model
final_model = None
if phase4_model:
    final_model = phase4_model
    checkpoint4 = os.path.join(MODELS_DIR, "te_dqn_phase4_extreme.pt")
    shutil.copy(phase4_model, checkpoint4)
    print(f"Phase 4 extreme model saved to: {checkpoint4}")
# elif phase3_model:
#     final_model = phase3_model
# elif phase2_model:
#     final_model = phase2_model
# else:
#     final_model = phase1_model

# Copy final model to output
final_path = os.path.join(MODELS_DIR, "te_dqn_curriculum_final.pt")
shutil.copy(final_model, final_path)
print(f"Final curriculum model saved to: {final_path}")

# Wait before stress test
print(f"Waiting {WAIT_TIME} seconds before stress test...")
time.sleep(WAIT_TIME)

# Kill SUMO processes before stress test
kill_sumo()

# Run stress test
print("Running stress test with final model...")
stress_test_dir = os.path.join(MODELS_DIR, "te_dqn_stress_test")
os.makedirs(stress_test_dir, exist_ok=True)

stress_cmd = [
    "python", "high-load-scenario.py",
    "--ff_model", "best_model/best_model_ff.pth",  # Assuming this exists
    "--te_model", final_path,
    "--intensity", "high",
    "--episodes", "2",
    "--output", stress_test_dir
]

print(f"Running command: {' '.join(stress_cmd)}")
stress_result = subprocess.run(stress_cmd)
print(f"Stress test completed with exit code: {stress_result.returncode}")

# Check for stress test results
for root, dirs, files in os.walk(stress_test_dir):
    for file in files:
        if file.endswith(".txt") or file.endswith(".json") or file.endswith(".png"):
            result_file = os.path.join(root, file)
            print(f"Found result file: {result_file}")

print(f"\nTE-DQN Curriculum Training Complete!")
print(f"Final model: {final_path}")
print(f"Stress test results: {stress_test_dir}")