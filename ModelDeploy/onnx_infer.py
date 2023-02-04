# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple
from modules.smoke_bbox_coder import SMOKECoder

import onnxruntime as onnxrt
import numpy as np
import time
import cv2

import torch
from torch import Tensor
from torch.nn import functional as F


class SmokeInfer:
    """SmokeInfer class.
    """

    def __init__(self,
                 model_path,
                 onnx_providers=None,
                 shared_library_path='/opt/onnxruntime/lib/libmmdeploy_onnxruntime_ops.so'
                 ):
        if onnx_providers is None:
            onnx_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.model_path = model_path

        self.bbox_code_size = 7
        self.bbox_coder = SMOKECoder(base_depth=(28.01, 16.32),
                                     base_dims=((0.88, 1.73, 0.67), (1.78, 1.70, 0.58), (3.88, 1.63, 1.53)),
                                     code_size=7)
        self.mean = [123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
        self.img_metas = [
            dict(
                cam2img=[[721.5377, 0.0, 609.5593, 44.85728],
                         [0.0, 721.5377, 172.854, 0.2163791],
                         [0.0, 0.0, 1.0, 0.002745884],
                         [0.0, 0.0, 0.0, 1.0]],
                trans_mat=np.array(
                    [[0.25, 0., 0.], [0., 0.25, 0], [0., 0., 1.]],
                    dtype=np.float32)
            )
        ]

        session_option = onnxrt.SessionOptions()
        session_option.register_custom_ops_library(shared_library_path)
        self.session = onnxrt.InferenceSession(self.model_path, sess_options=session_option,
                                               providers=onnx_providers)

    def warmup(self):
        for idx in range(20):
            inputs = np.random.rand(1, 3, 384, 1280).astype('f')
            start = time.time()
            _ = self.session.run(None, {"img": inputs})
            print(f"Iter {idx}: {time.time()- start:.5f}sec")
        print("WarmUp completed!")

    def predict(self, img):
        start = time.time()

        # Det3DDataPreprocessor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1280, 384))
        img = self.normalize_image(img, self.mean, self.std)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # Inference
        outputs = self.session.run(None, {"img": img})
        print(f"Session: {time.time() - start:.5f}sec")

        result = self.predict_by_feat([Tensor(outputs[0])],
                                      [Tensor(outputs[1])],
                                      self.img_metas)
        return result

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None) -> List:
        assert len(cls_scores) == len(bbox_preds) == 1
        cam2imgs = torch.stack([
            cls_scores[0].new_tensor(img_meta['cam2img'])
            for img_meta in batch_img_metas
        ])
        trans_mats = torch.stack([
            cls_scores[0].new_tensor(img_meta['trans_mat'])
            for img_meta in batch_img_metas
        ])
        batch_bboxes, batch_scores, batch_topk_labels = self._decode_heatmap(
            cls_scores[0],
            bbox_preds[0],
            batch_img_metas,
            cam2imgs=cam2imgs,
            trans_mats=trans_mats,
            topk=100,
            kernel=3)

        result_list = []
        for img_id in range(len(batch_img_metas)):

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

        center_heatmap_pred = SmokeInfer.get_local_maximum(cls_score, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = SmokeInfer.get_topk_from_heatmap(
            center_heatmap_pred, k=topk)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        regression = SmokeInfer.transpose_and_gather_feat(reg_pred, batch_index)
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
        feat = SmokeInfer.gather_feat(feat, ind)
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

    @staticmethod
    def normalize_image(img, mean, std):
        img = (img - mean) / std
        return img.astype('f')


def test():
    smoke = SmokeInfer(model_path="/home/admin/detection3d/mmdeploy/smoke/end2end.onnx")
    #smoke.warmup()

    img = cv2.imread("/home/admin/detection3d/mmdetection3d/data/kitti/training/image_2/000001.png")

    outputs = smoke.predict(img)
    print(outputs)


if __name__ == "__main__":
    test()
