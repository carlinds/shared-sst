#!/bin/bash

#SBATCH -A SNIC2022-5-184 -p alvis
#SBATCH --nodes 1
#SBATCH --gpus-per-node=V100:4
#SBATCH --time 3-00:00:00
#SBATCH --output /cephyr/users/carlinds/Alvis/slurm/%j.out
#SBATCH --no-requeue
#

echo $SLURM_JOB_ID

CONFIG=$1
GPUS_PER_NODE=4
REPO_NAME="shared-sst"
#SINGULARITY_IMAGE="shared-sst-latest.sif"
REPO_DIR="/cephyr/users/carlinds/Alvis"
DATASET_PATH="/mimer/NOBACKUP/groups/snic2021-7-127/eliassv/data"

# WANDB parameters
WANDB_ENTITY="carlinds"
WANDB_PROJECT="shared-sst"
WANDB_GROUP="lidar-only-image-grouping"
CONFIG_BASENAME=$(basename $CONFIG .py)
WANDB_NAME="${CONFIG_BASENAME}_${SLURM_JOB_ID}"
WANDB_JOB_TYPE="train"
WANDB_ID="${SLURM_JOB_ID}"

export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=12345
export PORT=$MASTER_PORT

singularity exec --nv \
    -B $REPO_DIR/$REPO_NAME:/$REPO_NAME \
    -B $DATASET_PATH:$DATASET_PATH \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --pwd /$REPO_NAME \
    $REPO_DIR/voxel-mae/voxel-mae-latest.sif \
    bash tools/dist_train.sh $CONFIG $GPUS_PER_NODE \
    --work-dir /$REPO_DIR/jobs/$SLURM_JOB_ID \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --wandb_group $WANDB_GROUP \
    --wandb_name $WANDB_NAME \
    --wandb_job_type $WANDB_JOB_TYPE \
    --wandb_id $WANDB_ID \
    --wandb_save_code

#
#EOF
