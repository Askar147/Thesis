#!/usr/bin/env python3
"""
This script wraps the original train_transformer_main.py
and automatically answers 'n' to the evaluation question
"""

import os
import sys
import subprocess
from unittest.mock import patch
import builtins

def main():
    # Get the original script arguments
    args = sys.argv[1:]
    
    if not args:
        print("Usage: python fix_train_transformer_main.py [arguments for train_transformer_main.py]")
        sys.exit(1)
    
    # Check if skip_evaluation is in the arguments
    skip_eval = False
    if "--skip_evaluation" in args:
        skip_eval = True
        # Remove it from the arguments as the original script doesn't recognize it
        args = [arg for arg in args if arg != "--skip_evaluation"]
    
    # Build the command to run the original script
    cmd = ["python", "train_transformer_main.py"] + args
    
    print(f"Running train_transformer_main.py with automatic input handling")
    print(f"Command: {' '.join(cmd)}")
    
    if skip_eval:
        # If skipping evaluation, we'll use mock to intercept input() calls
        # and automatically respond with 'n'
        with patch.object(builtins, 'input', lambda _: 'n'):
            result = subprocess.run(cmd)
    else:
        # Otherwise, just run it normally
        result = subprocess.run(cmd)
    
    # Return the same exit code
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()