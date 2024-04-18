#!/bin/bash
#SBATCH --job-name=baker_ph_2
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output baker_ph_2-%j.out

# Load any necessary modules or activate virtual environments
source /lab/barcheese01/mdiberna/OpticalPooledScreens/venv_ops_new/bin/activate

# Change to the directory where your Snakefiles are located
cd /lab/barcheese01/screens/baker

# Execute Snakemake
snakemake -s baker_ph_2.smk --use-conda --cores=all