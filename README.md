# Shared SST

## Usage
### Environment
**PyTorch >= 1.9 is recommended for a better support of the checkpoint technique.**
(or you can manually replace the interface of checkpoint in torch < 1.9 with the one in torch >= 1.9.)

The implementation builds upon code from [SST](https://github.com/TuSimple/SST), which in turn is based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d). Please refer to their [getting_started](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md) for getting the environment up and running.

### Setup
The docker file `docker/Dockerfile` contains the necessary environment for getting started.
It is recommended to build a container from this environment and then use that container to install all needed build dependencies and build the CUDA code by running:
```
pip install -r requirements/build.txt
pip install --no-cache-dir -e .
```

### Training models
The training procedure is the same as the one in SST. Please refer to `./tools/train.py` or `./tools/dist_train.sh` for details.

## Acknowledgments
This project is based on the following codebases.  

* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
* [LiDAR-RCNN](https://github.com/TuSimple/LiDAR_RCNN)
* [SST](https://github.com/TuSimple/SST)
