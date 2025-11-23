#!/bin/bash
#SBATCH --job-name=bio_analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --account=bioinf593f25_class
#SBATCH --partition=standard
#SBATCH --output=logs/bio_%j.log

module purge
module load python-anaconda3 2>/dev/null || module load anaconda 2>/dev/null
eval "$(conda shell.bash hook)"
CONDA_ENV_PATH="/gpfs/accounts/wilms_root/wilms0/jinhlee/CCC/conda_env"
conda activate "$CONDA_ENV_PATH"

echo "Running Biological Validation..."
python3 biological_validation.py
