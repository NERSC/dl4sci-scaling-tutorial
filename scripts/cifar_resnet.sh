#!/bin/bash
#SBATCH -J cifar-resnet
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -q regular
#SBATCH --reservation dl4sci
#SBATCH -t 1:00:00
#SBATCH -o logs/%x-%j.out

# Load the software
module load tensorflow/intel-1.13.1-py36
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"

# Ensure dataset is downloaded by single process
python -c "import keras; keras.datasets.cifar10.load_data()"

# Submit multi-node training
srun -u python train_horovod.py configs/cifar10_resnet.yaml -d
