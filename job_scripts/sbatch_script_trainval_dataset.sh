#!/usr/bin/env bash
#SBATCH -A SNIC2021-7-127 -p alvis
#SBATCH -t 16:00:00
#SBATCH --gpus-per-node=A40:4
#SBATCH -N 1
#SBATCH --output=/cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/slurm-out/slurm-%j.out
#SBATCH -J "Some job name"  # multi node, multi GPU
CONFIG=${1:-sst_nuscenesD5_1x_3class_8heads_v2}
REPO_NUMBER=${2:-1}  # Choose between repo 1 or 2
echo $CONFIG
GPUS_PER_NODE=4
export GPU_TYPE=A40
# Options for GPU type are T4 or A40. Choose A40 also when running on V100, A100 or A100fat.
export OMP_NUM_THREADS=16  # cores per gpu (16 cores per A40)


echo $HOSTNAME
echo $SLURM_JOB_NODELIST

export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=12345
export PORT=$MASTER_PORT
export JOB_ID=$SLURM_JOB_ID
export NGPUS_PER_NODE=$(echo "$SLURM_GPUS_PER_NODE" | sed 's/[A-Z0-9]*:\([0-9]*\)*/\1/')

echo ""
echo "This job $JOB_ID was started as:
  bash tools/dist_train.sh configs/sst_refactor/$CONFIG.py $GPUS_PER_NODE \
    --work-dir /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/jobs/$JOB_ID \
    --cfg-options evaluation.metric=nuscenes" ${@:3}
echo ""

echo ""
echo "Start copying repo to '$TMPDIR'"
if [ $REPO_NUMBER == 1 ]
then
   echo "Taking SST_${GPU_TYPE}"
   cp -r /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/SST_${GPU_TYPE}/ $TMPDIR/SST_${GPU_TYPE}
else
   echo "Taking SST_${GPU_TYPE}_2"
   cp -r /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/SST_${GPU_TYPE}_2/ $TMPDIR/SST_${GPU_TYPE}
fi
echo ""
echo "Copying of repo to tempdir is now done."
echo ""

echo ""
echo "Start copying nusenes info to '$TMPDIR/SST_$GPU_TYPE/'"
unzip -q -d $TMPDIR/SST_$GPU_TYPE /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/nuscenes_info/nuscenes_infos_v2.zip
echo ""
echo "Copying of nusenes info to repo in tempdir is now done."
echo ""

echo ""
echo "Start copying dataset to '$TMPDIR/SST_$GPU_TYPE/'"
source ./unpack_trainval.sh
echo ""
echo "Copying of dataset to repo in tempdir is now done."
echo ""

cd $TMPDIR/SST_$GPU_TYPE
singularity exec --pwd $TMPDIR/SST_$GPU_TYPE \
  /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/sst_env/mmdetection3d_$GPU_TYPE.sif \
  bash tools/dist_train.sh configs/sst_refactor/$CONFIG.py $GPUS_PER_NODE \
  --work-dir /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/jobs/$JOB_ID \
  --cfg-options evaluation.metric=nuscenes ${@:3}
# ${@:3} grabs everything after the third argument these are used to overwrite settings in the config file
# e.g. load_from="config/path/epoch_n.pth" loads the weights from the provided model

cp /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/slurm-out/slurm-$JOB_ID.out /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/jobs/$JOB_ID
