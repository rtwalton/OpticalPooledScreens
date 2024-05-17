#!/bin/bash
#SBATCH --job-name=baker_merge_3
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output baker_merge_3-%j.out

# Load any necessary modules or activate virtual environments
source /lab/barcheese01/mdiberna/OpticalPooledScreens/venv_ops_new/bin/activate

# Change to the directory where your Snakefiles are located
cd /lab/barcheese01/screens/baker

# Execute Snakemakes
snakemake -s baker_hash_3.smk --use-conda --cores=all

# Execute Snakemakes
snakemake -s baker_merge_3.smk --use-conda --cores=all