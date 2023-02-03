# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional, Tuple
import onnxruntime as onnxrt
import numpy as np
import time
import cv2
import torch
from torch import Tensor
from torch.nn import functional as F

from .. import modules as md

class ONNXSmoke:
    """
    원본 : SmokeInfer Class
     - 작성자 : 한상준
    수정 : ONNXSmoke
     - 수정자 : 김형석
     - 수정 내용 : Inference Engine에 적용하기 위해 일부 구조 변경 / 고정된 metadata 삭제, forward 입력 파라미터 변경

    """

    def __init__(self,
                 weight_path:str,
                 input_width:int = 1280,
                 input_height:int = 384,
                 shared_library_path:str='/opt/onnxruntime/lib/libmmdeploy_onnxruntime_ops.so',
                 onnx_providers:Optional[List[str]]=None,
                 ):
        
        if onnx_providers is None:
            onnx_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.__weight_path = weight_path
        self.__input_width = input_width
        self.__input_height = input_height
        self.__input_converter = md.InputConverter(input_width=input_width, 
                                                   input_height=input_height, 
                                                   input_type='ndarray')
        self.bbox_code_size = 7
        self.bbox_coder = md.SMOKECoder(base_depth=(28.01, 16.32),
                                     base_dims=((0.88, 1.73, 0.67), (1.78, 1.70, 0.58), (3.88, 1.63, 1.53)),
                                     code_size=7)
        session_option = onnxrt.SessionOptions()
        session_option.register_custom_ops_library(shared_library_path)
        self.session = onnxrt.InferenceSession(self.__weight_path, sess_options=session_option,
                                               providers=onnx_providers)

    def warmup(self):
        for idx in range(20):
            inputs = np.random.rand(1, 3, self.__input_height, self.__input_width).astype('f')
            start = time.time()
            _ = self.session.run(None, {"img": inputs})
            print(f"Iter {idx}: {time.time()- start:.5f}sec")
        print("WarmUp completed!")

    def forward(self, image:np.ndarray, meta_data:List[Dict[str, Any]]) -> md.InferenceResult:
        input_data = self.__input_converter(image)
        start = time.time()
        data = {
            "img": input_data  # convert image dimention to (1,C,H,W)
        }

        # Inference
        outputs = self.session.run(None, data)
        print(f"Session: {time.time() - start:.5f}sec")

        results = self.predict_by_feat(Tensor(outputs[0]), Tensor(outputs[1]), meta_data)
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
            batch_img_metas, # type: ignore
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
                        batch_img_metas: List[dict],
                        cam2imgs: Tensor,
                        trans_mats: Tensor,
                        topk: int = 100,
                        kernel: int = 3) -> Tuple[Tensor, Tensor, Tensor]:
        bs, _, feat_h, feat_w = cls_score.shape

        center_heatmap_pred = ONNXSmoke.get_local_maximum(cls_score, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = ONNXSmoke.get_topk_from_heatmap(
            center_heatmap_pred, k=topk)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        regression = ONNXSmoke.transpose_and_gather_feat(reg_pred, batch_index)
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
        feat = ONNXSmoke.gather_feat(feat, ind)
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
