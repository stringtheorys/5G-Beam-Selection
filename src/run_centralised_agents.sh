#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

module load conda
# conda create -n 5GBeamAlignment python=3.7
# pip install tensorflow-federated
source activate 5GBeamAlignment

module load cuda/11.0

echo $PWD
cd ~/5G-Beam-Selection/src/

python main.py --agent centralised --model imperial
python main.py --agent centralised --model beamsoup-lidar
python main.py --agent centralised --model beamsoup-coord
python main.py --agent centralised --model beamsoup
