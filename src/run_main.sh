#!/bin/bash

#SBATCH --partition=gtx1080
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

module load conda
source activate 5GBeamAlignment
echo -e "import sys\nprint(sys.version)\nimport tensorflow as tf" | python

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
