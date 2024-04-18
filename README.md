## Optical Pooled Screens of Essential Genes

This repository contains code and computational tools related to the publication, [*The phenotypic landscape of essential human genes*](https://pubmed.ncbi.nlm.nih.gov/36347254/).

For new projects using optical pooled screens, it is highly recommended to use the Github repository accompanying our Nature Protocols paper, [*Pooled genetic perturbation screens with image-based phenotypes*](https://pubmed.ncbi.nlm.nih.gov/35022620/): https://github.com/feldman4/OpticalPooledScreens.

## More about this repository

This repository contains additional application-specific resources for our study of essential gene function using optical pooled screens.

This includes:
- Many additional image features implemented as functions operating on scikit-image RegionProps objects (features come from CellProfiler and additional sources)
- Functions for analyzing live-cell optical pooled screens (calling TrackMate for cell tracking)

## Installation (OSX)

**WARNING: many versions of dependencies will have trouble installing on Python 3.8. It is currently recommended to use Python 3.6. Setting up a Python 3.6 conda environment may be a convenient solution, set-up guide [here](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-python).**

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

The triangle hash and merging steps will be added soon, along with corresponding Jupyter notebooks.

