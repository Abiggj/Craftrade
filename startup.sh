#!/bin/bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda init bash
source ~/.bashrc

# Create a conda environment and activate it
conda create -n myenv python=3.8 -y
conda activate myenv

# Install requirements
conda install -r requirements.txt

# Start the Flask app
python server.py
