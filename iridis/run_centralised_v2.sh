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

python main.py --agent centralised-v2 --model imperial
python main.py --agent centralised-v2 --model beamsoup-coord
python main.py --agent centralised-v2 --model southampton
