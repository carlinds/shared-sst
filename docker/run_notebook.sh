#!/bin/sh

DATA_PATH=/datasets/nuscenes
REPO_ROOT=~/repos/github
REPO_NAME=shared-sst
DOCKER_IMAGE=shared-sst:develop

docker run --gpus all --shm-size=8g -p 8888:8888 \
-v $DATA_PATH:$DATA_PATH \
-v $REPO_ROOT/$REPO_NAME:/$REPO_NAME $DOCKER_IMAGE \
/bin/bash -i -c "cd /shared-sst; pip install --no-cache-dir -e .; jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"