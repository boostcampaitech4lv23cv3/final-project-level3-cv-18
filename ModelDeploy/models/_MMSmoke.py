# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from time import sleep
from typing import Any, Dict, List, Optional, Tuple
from mmdet3d.utils.typing_utils import InstanceList
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet3d.utils import register_all_modules, replace_ceph_backend
from mmdet3d.models.detectors import SMOKEMono3D
import math
import torch
from torch import Tensor
from . import ModelBase
from .. import modules as md

__all__ = ['MMSmoke']

class MMSmoke(ModelBase):
    """
    ## MMSmoke(ModelBase)
    이미지를 입력받아서 3D Object detection을 결과를 반환해줍니다.
    내부적으로 MMLab의 MMDet3D Engine으로 Inference됩니다.

    Examples:
        >>> model = MMSmoke()
        >>> image = cv2.imread(some_path)
        >>> inference_result = model.forward(image)
    Author : 김형석
    """
    def __init__(self, 
                 weight_path:str = "mmdetection3d/checkpoints/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth",
                 input_width:int = 1280,
                 input_height:int = 384, 
                 config_path:str = './ModelDeploy/models/mmconfig/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d.py') -> None:
        super().__init__(weight_path, input_width, input_height, 'tensor')
        self.__config_path = config_path
        register_all_modules(init_default_scope=False)
        self.__cfg = Config.fromfile(config_path)
        self.__cfg.launcher = 'none'
        self.__cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config_path))[0])
        self.__cfg.load_from = weight_path
        self.__runner = Runner.from_cfg(self.__cfg)
        self.__runner.load_or_resume()
        self.__model:SMOKEMono3D = self.__runner.model # type: ignore    
        self.__model.eval()


    def _forward(self, input_data, meta_data:List[Dict[str, Any]]) -> md.InferenceResult:
        input_data = input_data.to('cuda')
        data = {
            "imgs": input_data  # convert image dimention to (1,C,H,W)
        }
        cls_scores, bbox_preds = self.__model.forward(inputs=data, mode='tensor') # type: ignore   
        pred = self.__model.bbox_head.predict_by_feat(cls_scores, bbox_preds, meta_data)
        scores:torch.Tensor = pred[0].scores_3d
        labels:torch.Tensor = pred[0].labels_3d
        bboxes:torch.Tensor = pred[0].bboxes_3d.tensor
        return md.InferenceResult(bboxes, labels, scores) # type: ignore