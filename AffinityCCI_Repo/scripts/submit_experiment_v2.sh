#!/bin/bash
#SBATCH --job-name=affinity_exp_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --account=bioinf593f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/exp_v2_%j.log

module purge
module load python-anaconda3 2>/dev/null || module load anaconda 2>/dev/null
eval "$(conda shell.bash hook)"

# Use your existing conda env
CONDA_ENV_PATH="/scratch/wilms_root/wilms0/jinhlee/CCC/conda_env"
conda activate "$CONDA_ENV_PATH"

echo "================================================================================"
echo "STARTING ADVANCED METRICS EXPERIMENT"
echo "================================================================================"

echo "[Phase 1] Training..."
python3 train_experiment_v2.py

if [ $? -eq 0 ]; then
    echo ""
    echo "[Phase 2] Plotting..."
    python3 visualize_v2.py
else
    echo "Training failed."
    exit 1
fi

echo "================================================================================"
echo "DONE"
