## Optical Pooled Screens of Essential Genes

## More about this repository

This repository is currently undergoing restructuring and a more complete README and walkthrough is on the horizon. Please be patient as we pressure test this across upcoming screens!

## Installation (OSX)

Download the OpticalPooledScreens directory (e.g., on Github use the green "Clone or download" button, then "Download ZIP").

In Terminal, go to the OpticalPooledScreens project directory and create a Python 3 virtual environment using a command like:

```bash
python3 -m venv venv_ops_new
```
This creates a virtual environment called `venv` for project-specific resources. The commands in `install.sh` add required packages to the virtual environment:

```bash
sh install.sh
```

The `ops` package is installed with `pip install -e`, so the source code in the `ops/` directory can be modified in place.

Once installed, activate the virtual environment from the project directory:

```bash
source venv/bin/activate
```

This will install based on the `requirements_venv_ops_new_18_04_2024.txt`, which has not fully been tried, but is up and running on the Whitehead clusters. The previous `requirements.txt` file is what was used in Luke's original codebase.

## Running

The `example_analysis/` directory contains the majorit of the required code to run the analysis. Within each subfolder, there are Jupyter notebooks that accompany the pre-processing, sbs, and ph steps of the analysis.
