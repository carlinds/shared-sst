#!/bin/bash

#SBATCH -A SNIC2022-5-184 -p alvis
#SBATCH --nodes 1
#SBATCH --gpus-per-node=V100:4
#SBATCH --time 3-00:00:00
#SBATCH --output /cephyr/users/carlinds/Alvis/slurm/%j.out
#SBATCH --no-requeue
#

CONFIG=$1
GPUS_PER_NODE=4
REPO_NAME="voxel-mae"
SINGULARITY_IMAGE="voxel-mae-latest.sif"
REPO_DIR="/cephyr/users/carlinds/Alvis"
DATASET_PATH="/mimer/NOBACKUP/groups/snic2021-7-127/eliassv/data"

export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=12345
export PORT=$MASTER_PORT

singularity exec --nv \
    -B $REPO_DIR/$REPO_NAME:/$REPO_NAME \
    -B $DATASET_PATH:$DATASET_PATH \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --pwd /$REPO_NAME \
    $REPO_DIR/$REPO_NAME/$SINGULARITY_IMAGE \
    bash tools/dist_train.sh $CONFIG $GPUS_PER_NODE

#
#EOF
