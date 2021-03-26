#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

module load conda
# conda create -n 5GBeamAlignment python=3.7
# pip install tensorflow-federated
source activate 5GBeamAlignment

module load cuda/11.0

echo $PWD
cd ~/5G-Beam-Selection/src/
PYTHONPATH=~/5G-Beam-Selection/src/

python -m agents.centralised.py --model imperial
python -m agents.centralised.py --model beamsoup-lidar
python -m agents.centralised.py --model beamsoup-coord
python -m agents.centralised.py --model beamsoup
