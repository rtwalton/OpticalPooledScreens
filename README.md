# Optical Pooled Screens

## About this Repository

This repository is currently undergoing restructuring. A more complete README and walkthrough are coming soon. We appreciate your patience as we test this across upcoming screens.

## Installation (OSX)

### Step 1: Download OpticalPooledScreens Directory

```sh
git clone https://github.com/cheeseman-lab/OpticalPooledScreens.git
```

### Step 2: Install Python 3 Virtual Environment

```sh
# Navigate to the project directory
cd OpticalPooledScreens

# Create virtual environment and install packages
sh install.sh
```

This creates a virtual environment called `venv_ops_new` for project-specific resources and adds required packages to the virtual environment.

The `ops` package is installed with `pip install -e`, so the source code in the `ops/` directory can be modified in place.

Once installed, one can activate the virtual environment from the project directory with

```sh
source venv_ops/bin/activate
```

Note: This installation uses `requirements_ops.txt`. The previous `requirements.txt` is from the original codebase.

## Running

The `example_analysis/` directory contains the main code for running the analysis. Each subfolder includes Jupyter notebooks for the pre-processing, SBS, and PH steps of the analysis.

We're working on improving this documentation. Thank you for your understanding.
