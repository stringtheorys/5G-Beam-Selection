#!/bin/bash

#SBATCH --partition=gtx1080
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

module load conda
# conda create -n 5GBeamAlignment python=3.6
# pip install tensorflow
conda activate 5GBeamAlignment

module load cuda/11.0

echo $PWD
cd ~/5G-Beam-Selection/src/

python main.py --agent centralised --model imperial

python main.py --agent centralised --model beamsoup-lidar
python main.py --agent centralised --model beamsoup-coord
python main.py --agent centralised --model beamsoup-joint

python main.py --agent centralised --model husky-fusion

python main.py --agent centralised --model southampton

python main.py --agent southampton --model imperial
python main.py --agent southampton --model southampton

python main.py --agent federated --model imperial
python main.py --agent federated --model beamsoup-coord
python main.py --agent distributed --model imperial
python main.py --agent distributed --model beamsoup-coord