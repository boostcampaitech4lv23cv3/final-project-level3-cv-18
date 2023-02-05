# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import time
import cv2
import ctypes

import torch
from torch import Tensor
from torch.nn import functional as F

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

from . import ModelBase
from .. import modules as md


__all__ = ['TRTSmoke', 'HostDeviceMem']


# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTSmoke(ModelBase):
    """
    원본 : ONNXSmoke Class
     - 작성자 : 김형석
    수정 : TRTSmoke
     - 수정자 : 한상준
     - 수정 내용 : ONNX → TensorRT

    """

    def __init__(self,
                 weight_path:str,
                 input_width:int = 1280,
                 input_height:int = 384,
                 shared_library_path:str='./ModelDeploy/lib/libmmdeploy/libmmdeploy_trt_net.so'
                 ):
        super().__init__(weight_path, input_width, input_height, 'ndarray')
        self.shared_library_path = shared_library_path
        self.bbox_code_size = 7
        self.bbox_coder = md.SMOKECoder(base_depth=(28.01, 16.32),
                                     base_dims=((0.88, 1.73, 0.67), (1.78, 1.70, 0.58), (3.88, 1.63, 1.53)),
                                     code_size=7)

        ### Load TensorRT Model
        ctypes.CDLL(self.shared_library_path)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        self.trt_runtime = trt.Runtime(TRT_LOGGER)

        with open(self.__weight_path, 'rb') as f:
            trt_model = f.read()
        self.engine = self.trt_runtime.deserialize_cuda_engine(trt_model)
        self.context = self.engine.create_execution_context()

        ### Memory allocation
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT:
                is_input = True
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def cls_score_spec(self):
        return self.outputs[0]["shape"], self.outputs[0]["dtype"]

    def bbox_pred_spec(self):
        return self.outputs[1]["shape"], self.outputs[1]["dtype"]

    def warmup(self):
        for idx in range(4):
            inputs = np.random.rand(1, 3, self.__input_height, self.__input_width).astype('f')
            start = time.time()
            cls_score = np.zeros(*self.cls_score_spec())
            bbox_pred = np.zeros(*self.bbox_pred_spec())
            cuda.memcpy_htod(self.inputs[0]["allocation"], np.ascontiguousarray(np.random.rand(1, 3, 384, 1280)))
            self.context.execute_v2(self.allocations)
            cuda.memcpy_dtoh(cls_score, self.outputs[0]["allocation"])
            cuda.memcpy_dtoh(bbox_pred, self.outputs[1]["allocation"])
            print(f"Iter {idx}: {time.time()- start:.5f}sec")
        print("WarmUp completed!")

    def _forward(self, image:np.ndarray, meta_data:List[Dict[str, Any]]) -> md.InferenceResult:
        input_data = self.__input_converter(image)
        start = time.time()

        # Inference
        cls_score = np.zeros(*self.cls_score_spec())
        bbox_pred = np.zeros(*self.bbox_pred_spec())
        cuda.memcpy_htod(self.inputs[0]["allocation"], np.ascontiguousarray(input_data))
        self.context.execute_v2(self.allocations)
        cuda.memcpy_dtoh(cls_score, self.outputs[0]["allocation"])
        cuda.memcpy_dtoh(bbox_pred, self.outputs[1]["allocation"])
        print(f"Session: {time.time() - start:.5f}sec")

        results = self.predict_by_feat(Tensor(cls_score), Tensor(bbox_pred), meta_data)
        result = results[0]
        print("result = ", result)
        scores:torch.Tensor = result['scores_3d']
        labels:torch.Tensor = result['labels_3d']
        bboxes:torch.Tensor = result['bboxes_3d']
        return md.InferenceResult(bboxes, labels, scores) # type: ignore

    def predict_by_feat(self, cls_score:Tensor, bbox_pred:Tensor,
                        batch_img_metas: Optional[List[dict]] = None) -> List:
        cls_scores = [cls_score]
        bbox_preds = [bbox_pred]
        assert len(cls_scores) == len(bbox_preds) == 1
        cam2imgs = torch.stack([
            cls_scores[0].new_tensor(img_meta['cam2img'])
            for img_meta in batch_img_metas # type: ignore
        ])
        trans_mats = torch.stack([
            cls_scores[0].new_tensor(img_meta['trans_mat'])
            for img_meta in batch_img_metas # type: ignore
        ])
        batch_bboxes, batch_scores, batch_topk_labels = self._decode_heatmap(
            cls_scores[0],
            bbox_preds[0],
            cam2imgs=cam2imgs,
            trans_mats=trans_mats,
            topk=100,
            kernel=3)

        result_list = []
        for img_id in range(len(batch_img_metas)): # type: ignore

            bboxes = batch_bboxes[img_id]
            scores = batch_scores[img_id]
            labels = batch_topk_labels[img_id]

            keep_idx = scores > 0.25
            bboxes = bboxes[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]

            for bbox in bboxes:
                bbox[1] += (bbox[4] / 2)

            results = dict(
                bboxes_3d=bboxes,
                labels_3d=labels,
                scores_3d=scores
            )
            result_list.append(results)
        return result_list

    def _decode_heatmap(self,
                        cls_score: Tensor,
                        reg_pred: Tensor,
                        cam2imgs: Tensor,
                        trans_mats: Tensor,
                        topk: int = 100,
                        kernel: int = 3) -> Tuple[Tensor, Tensor, Tensor]:
        bs, _, feat_h, feat_w = cls_score.shape

        center_heatmap_pred = TRTSmoke.get_local_maximum(cls_score, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = TRTSmoke.get_topk_from_heatmap(
            center_heatmap_pred, k=topk)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        regression = TRTSmoke.transpose_and_gather_feat(reg_pred, batch_index)
        regression = regression.view(-1, 8)

        points = torch.cat([topk_xs.view(-1, 1),
                            topk_ys.view(-1, 1).float()],
                           dim=1)
        locations, dimensions, orientations = self.bbox_coder.decode(
            regression, points, batch_topk_labels, cam2imgs, trans_mats)

        batch_bboxes = torch.cat((locations, dimensions, orientations), dim=1)
        batch_bboxes = batch_bboxes.view(bs, -1, self.bbox_code_size)
        return batch_bboxes, batch_scores, batch_topk_labels

    @staticmethod
    def get_local_maximum(heat, kernel=3):
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    @staticmethod
    def get_topk_from_heatmap(scores, k=20):
        batch, _, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
        topk_clses = topk_inds // (height * width)
        topk_inds = topk_inds % (height * width)
        topk_ys = topk_inds // width
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    @staticmethod
    def transpose_and_gather_feat(feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = TRTSmoke.gather_feat(feat, ind)
        return feat

    @staticmethod
    def gather_feat(feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).repeat(1, 1, dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat