#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=clara

# Load necessary os level modules
module load Python/3.8
module load CUDA
module load libsndfile/1.0.28-GCCcore-10.2.0

# Load conda environment (here the python level - pip requirements should be installed to)
source venv/bin/activate

venv/bin/python run_model.py