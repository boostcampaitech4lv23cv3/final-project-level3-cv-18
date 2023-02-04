# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple
from modules.smoke_bbox_coder import SMOKECoder
#from mmdeploy.backend.tensorrt import load

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

class SmokeInferTRT:
    """
    SmokeInferTRT class.
    """

    def __init__(self,
                 model_path,
                 shared_library_path='/data/home/lob/detection3d/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so'
                 ):
        self.model_path = model_path
        self.shared_library_path = shared_library_path

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

        ### Load TensorRT Model
        ctypes.CDLL(self.shared_library_path)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        #self.engine = load(model_path)

        """
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network,\
                builder.create_builder_config() as config, trt.OnnxParser(network,TRT_LOGGER) as parser,\
                trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 30  # 1G
            builder.max_batch_size = 1
            with open('../mmdeploy/smoke_trt/end2end.onnx', 'rb') as fr:
                if not parser.parse(fr.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    assert False

            print("Start to build Engine")
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            plan = engine.serialize()
            savepth = './model.trt'
            with open(savepth, "wb") as fw:
                fw.write(plan)
        model_path = savepth
        """

        with open(model_path, 'rb') as f:
            trt_model = f.read()
        self.engine = self.trt_runtime.deserialize_cuda_engine(trt_model)
        self.context = self.engine.create_execution_context()

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
        pass
        """
        for idx in range(20):
            inputs = np.random.rand(1, 3, 384, 1280).astype('f')
            start = time.time()
            _ = self.session.run(None, {"img": inputs})
            print(f"Iter {idx}: {time.time()- start:.5f}sec")
        print("WarmUp completed!")
        """

    def predict(self, img):
        # Det3DDataPreprocessor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1280, 384))
        img = self.normalize_image(img, self.mean, self.std)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # Inference
        cls_score = np.zeros(*self.cls_score_spec())
        bbox_pred = np.zeros(*self.bbox_pred_spec())
        cuda.memcpy_htod(self.inputs[0]["allocation"], np.ascontiguousarray(self.inputs[0]))
        self.context.execute_v2(self.allocations)
        cuda.memcpy_dtoh(cls_score, self.outputs[0]["allocation"])
        cuda.memcpy_dtoh(bbox_pred, self.outputs[1]["allocation"])

        result = self.predict_by_feat([Tensor(cls_score)],
                                      [Tensor(bbox_pred)],
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
            cam2imgs=cam2imgs,
            trans_mats=trans_mats,
            topk=100,
            kernel=3)

        result_list = []
        for img_id in range(len(batch_img_metas)):

            bboxes = batch_bboxes[img_id]
            scores = batch_scores[img_id]
            labels = batch_topk_labels[img_id]

            #print(scores)
            #quit()

            #keep_idx = scores > 0.25
            keep_idx = scores > 0.08
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

        center_heatmap_pred = SmokeInferTRT.get_local_maximum(cls_score, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = SmokeInferTRT.get_topk_from_heatmap(
            center_heatmap_pred, k=topk)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        regression = SmokeInferTRT.transpose_and_gather_feat(reg_pred, batch_index)
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
        feat = SmokeInferTRT.gather_feat(feat, ind)
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

    @staticmethod
    def allocate_buffers(engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        binding_to_type = {"img": np.float32, "cls_score": np.float32, "bbox_pred": np.float32}

        for binding in engine:
            size = trt.volume(engine.get_tensor_shape(binding)) * 1 #engine.max_batch_size
            dtype = binding_to_type[str(binding)]
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.get_tensor_mode(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream


def test():
    smoke = SmokeInferTRT(model_path="/data/home/lob/detection3d/ModelDeploy/models/smoke.engine")
    #smoke.warmup()

    for i in range(7000):
        img_filename = f"/data/home/lob/detection3d/mmdetection3d/data/kitti/training/image_2/{i:06}.png"
        print(img_filename)
        img = cv2.imread(img_filename)
        start = time.time()
        outputs = smoke.predict(img)
        print(f"Session: {time.time() - start:.5f}sec - {outputs}")

if __name__ == "__main__":
    test()
