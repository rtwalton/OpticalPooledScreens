#!/bin/bash
#SBATCH --job-name=merge_3_eval
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=500G
#SBATCH --time=2:00:00
#SBATCH --output out/merge_3_eval-%j.out

# Load any necessary modules or activate virtual environments
source /lab/barcheese01/mdiberna/OpticalPooledScreens/venv_ops_new/bin/activate

# Change to the directory where your Snakefiles are located
cd /lab/barcheese01/screens/denali

# Execute Snakemake
python3 merge_3/merge_3_eval.py