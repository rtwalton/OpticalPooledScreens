# Optical Pooled Screens

## About this Repository

This repository contains the computational pipeline for analyzing Optical Pooled Screens (OPS), a high-throughput imaging-based method for functional genomic screening. The codebase enables automated analysis of large-scale microscopy data to evaluate phenotypic changes resulting from CRISPR-based genetic perturbations.
Key features include:

- Processing and analysis of multi-round imaging data
- Cell segmentation and tracking across imaging rounds
- Guide RNA barcode decoding from fluorescence images
- Phenotypic feature extraction and quantification
- Statistical analysis of gene-phenotype relationships

The code is designed to handle the complex computational challenges of linking genetic perturbations to cellular phenotypes in pooled optical screens.

## Usage

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
This creates a virtual environment called `venv_ops` for project-specific resources and adds required packages to the virtual environment.
The `ops` package is installed with `pip install -e`, so the source code in the `ops/` directory can be modified in place.

Once installed, one can activate the virtual environment from the project directory with
```sh
source venv_ops/bin/activate
```

## Step 3: Running
The `example_analysis/` directory contains the main code for running OPS analysis. Each subfolder includes individual READMEs that specify how to do the analysis.
We are working on improving this codebase and a streamlined, optimized release is forthcoming.

## Issues
If you encounter any problems with the code, please open an issue on our GitHub repository.

We actively monitor issues and will work to resolve them as quickly as possible.

## Citations

### Primary Citation
If you use this code in your research, please cite:

Funk, L., Su, K. C., Ly, J., Feldman, D., Singh, A., Moodie, B., Blainey, P. C., & Cheeseman, I. M. (2022). The phenotypic landscape of essential human genes. Cell, 185(24), 4634-4653.e22. https://doi.org/10.1016/j.cell.2022.10.017

### Code Attribution
This codebase is adapted from the original OpticalPooledScreens repository:
https://github.com/lukebfunk/OpticalPooledScreens