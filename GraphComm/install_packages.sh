#!/bin/bash

set -e  # Exit immediately if any command fails

# Install JupyterLab
echo "Installing JupyterLab..."
conda install -y --channel=conda-forge jupyterlab==3.5.3
conda clean -ya

# Install Python packages with pip
echo "Installing Python packages..."
pip install -U --no-cache-dir \
    anndata==0.8.0 \
    liana==0.1.5\
    matplotlib==3.6.3 \
    numpy==1.22.0 \
    omnipath==1.0.6 \
    pandas==1.3.5 \
    rdflib==6.2.0 \
    scanpy==1.9.1 \
    scprep==1.2.2 \
    torch==1.13.1
#    torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# liana==0.1.5 

# Install additional packages with pip3
echo "Installing additional packages..."
pip3 install -U --no-cache-dir \
    leidenalg==0.9.1 

# # Install torch-scatter and torch-sparse
# echo "Installing torch-scatter and torch-sparse..."
# pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1.html
# pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.1.html
conda install pytorch-scatter -c pyg
conda install pytorch-sparse -c pyg

# Install PyTorch Geometric
echo "Installing PyTorch Geometric"
# pip install git+https://github.com/pyg-team/pytorch_geometric.git
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

conda install pyg=*=*cu* -c pyg

# Install PyTorch cluster
echo "Installing PyTorch cluster..."
conda install pytorch-cluster -c pyg

# Exit message
echo "All installations completed successfully!"

