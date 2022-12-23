#!/bin/sh

DATA_PATH=/datasets/nuscenes
REPO_ROOT=~/repos/github
REPO_NAME=shared-sst
DOCKER_IMAGE=shared-sst:develop

docker run --gpus all --shm-size=8g -p 8888:8888 -it \
-v $DATA_PATH:$DATA_PATH \
-v $REPO_ROOT/$REPO_NAME:/$REPO_NAME $DOCKER_IMAGE