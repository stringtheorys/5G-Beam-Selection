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

python main.py --agent federated --model imperial --vehicle 1
python main.py --agent federated --model imperial --vehicle 2
python main.py --agent federated --model imperial --vehicle 4
python main.py --agent federated --model imperial --vehicle 6
python main.py --agent federated --model imperial --vehicle 8

python main.py --agent federated --model beamsoup-coord --vehicle 1
python main.py --agent federated --model beamsoup-coord --vehicle 2
python main.py --agent federated --model beamsoup-coord --vehicle 4
python main.py --agent federated --model beamsoup-coord --vehicle 6
python main.py --agent federated --model beamsoup-coord --vehicle 8

python main.py --agent federated --model southampton --vehicle 1
python main.py --agent federated --model southampton --vehicle 2
python main.py --agent federated --model southampton --vehicle 4
python main.py --agent federated --model southampton --vehicle 6
python main.py --agent federated --model southampton --vehicle 8
