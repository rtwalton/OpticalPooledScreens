#!/bin/bash
#SBATCH --job-name=baker_preprocessing_0_multi_ph
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output baker_preprocessing_0_multi_ph-%j.out

# Load any necessary modules or activate virtual environments
source /lab/barcheese01/mdiberna/OpticalPooledScreens_david/venv/bin/activate

# Change to the directory where your Snakefiles are located
cd /lab/barcheese01/screens/baker/preprocessing_0

# Execute Snakemake
python3 preprocessing_multi_ph.py