import subprocess
import os
import json

# Create a more complete config file
# Run with the complete config
subprocess.run([
    "python", "vec-high-load-evaluation.py",
    "--mode", "single", 
    "--model_path", "./best_model",
    "--config_path", "./config.json",
    "--energy_csv", "./merged_dag1.csv",
    "--output_dir", "./high_load_results"
])

print("Evaluation complete!")