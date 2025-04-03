#!/usr/bin/env python3
"""
LLM-based Training Data Generator for SPLADE Model Fine-tuning

This script uses GPT models to generate high-quality training data for SPLADE model fine-tuning.
It processes documents, chunks them, and generates query-document pairs that can be used for training.

Usage:
    python generate.py --input-dir /path/to/documents --output-file training_data.json
"""

import os
import sys

from src.main import main

if __name__ == "__main__":
    # Add the current directory to the path so we can import the modules
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    main()
