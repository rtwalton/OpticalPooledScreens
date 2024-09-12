#!/usr/bin/env bash

# Create a new virtual environment called 'venv_ops'
python3 -m venv venv_ops

# Activate the 'venv_ops' virtual environment
. venv_ops/bin/activate

# Install wheel package to support binary installations
pip install wheel

# Install dependencies from the requirements file
pip install -r requirements_ops.txt

# Link ops package instead of copying
# Jupyter and Snakemake will import code from .py files in the ops/ directory
pip install -e .

echo "Installation complete. Virtual environment 'venv_ops' is set up."
