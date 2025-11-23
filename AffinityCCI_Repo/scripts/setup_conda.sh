#!/bin/bash

# 1. Load the Anaconda Module
# This allows us to use the 'conda' command
module purge
module load python-anaconda3

echo "Setting up Conda Environment in: $(pwd)/conda_env"

# 2. Initialize Conda for this script
# This magic line makes 'conda activate' work inside a script
eval "$(conda shell.bash hook)"

# 3. Create the environment locally (in your scratch folder)
# We use Python 3.11 and install pip right away
conda create -p ./conda_env python=3.11 pip -y

# 4. Activate the new environment
conda activate ./conda_env

# 5. Install PyTorch and PyG via CONDA (Crucial Step)
# Conda channels provide binaries compatible with older cluster OSs
echo "Installing PyTorch & GNN libraries..."
conda install pytorch==2.5.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install pyg -c pyg -y

# 6. Install standard data libraries via Pip
# (Safe to use pip for these simple packages)
echo "Installing pandas, scanpy, etc..."
pip install pandas numpy scikit-learn matplotlib scanpy h5py

echo "========================================"
echo "VERIFICATION:"
python3 -c "import torch; print(f'Torch: {torch.__version__}')"
python3 -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
echo "========================================"
