#!/bin/bash
#SBATCH --job-name=baker_downstream_4
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH --output baker_downstream_4-%j.out

# Load any necessary modules or activate virtual environments
source /lab/barcheese01/mdiberna/OpticalPooledScreens/venv_ops_new/bin/activate

# Change to the directory where your Snakefiles are located
cd /lab/barcheese01/screens/baker

# Execute Snakemake
python3 baker_downstream_4.py