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
from .. import modules as md


class MMSmoke:
    def __init__(self, checkpoint_path:str) -> None:
        register_all_modules(init_default_scope=False)
        self.config_path = './ModelDeploy/models/mmconfig/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d.py'
        self.cfg = Config.fromfile(self.config_path)
        self.cfg.launcher = 'none'
        self.cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(self.config_path))[0])
        self.cfg.load_from = checkpoint_path
        self.runner = Runner.from_cfg(self.cfg)
        self.runner.load_or_resume()
        self.model:SMOKEMono3D = self.runner.model # type: ignore    
        self.model.eval()


    def forward(self, input_data:torch.Tensor, meta_data:List[Dict[str, Any]]) -> md.InferenceResult:
        data = {
            "imgs": input_data.unsqueeze(dim=0)  # convert image dimention to (1,C,H,W)
        }
        cls_scores, bbox_preds = self.model.forward(inputs=data, mode='tensor') # type: ignore   
        pred = self.model.bbox_head.predict_by_feat(cls_scores, bbox_preds, meta_data)
        scores:torch.Tensor = pred[0].scores_3d
        labels:torch.Tensor = pred[0].labels_3d
        bboxes:torch.Tensor = pred[0].bboxes_3d.tensor
        return md.InferenceResult(bboxes, labels, scores) # type: ignore