# Copyright (c) OpenMMLab. All rights reserved.
#from .voxel_detection import VoxelDetection
#from .voxel_detection_model import VoxelDetectionModel
from .smoke_detection import MMDetection3d, SmokeDetection
from .smoke_detection_model import SmokeDetectionModel

__all__ = ['MMDetection3d', #'VoxelDetection', 'VoxelDetectionModel',
           'SmokeDetection', 'SmokeDetectionModel']
