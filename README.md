# Optical Pooled Screens

## About this Repository

This repository is currently undergoing restructuring. A more complete README and walkthrough are coming soon. We appreciate your patience as we test this across upcoming screens.

## Installation (OSX)
1. Download the OpticalPooledScreens directory.
2. In Terminal, navigate to the project directory and create a Python 3 virtual environment:

```
python3 -m venv venv_ops_new
```

This creates a virtual environment called `venv_ops_new` for project-specific resources. The commands in `install.sh` add required packages to the virtual environment:

```
sh install.sh
```

The `ops` package is installed with `pip install -e`, so the source code in the `ops/` directory can be modified in place.

Once installed, activate the virtual environment from the project directory:

```bash
source venv_ops/bin/activate
```

Note: This installation uses `requirements_venv_ops_new_18_04_2024.txt`. The previous `requirements.txt` is from the original codebase.

## Running

The `example_analysis/` directory contains the main code for running the analysis. Each subfolder includes Jupyter notebooks for the pre-processing, SBS, and PH steps of the analysis.

We're working on improving this documentation. Thank you for your understanding.
