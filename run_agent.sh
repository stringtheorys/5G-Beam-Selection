#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

module load conda
# conda create -n 5GBeamAlignment python=3.7
# pip install tensorflow-federated
conda activate 5GBeamAlignment

cd 5G-Beam-Alignment
python centralised_agent.py
python federated_agent.py