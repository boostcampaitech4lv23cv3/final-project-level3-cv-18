# PyTorch Model 변환
1. MMDeploy Installation
```
export ONNXRUNTIME_DIR=/usr/local
export TENSORRT_DIR=/usr/include/aarch64-linux-gnu # x86_84-linux-gnu
export CUDNN_DIR=/lib/aarch64-linux-gnu
export INSTALL_PREFIX=/nvme/opt/mmdeploy mkdir build -p && cd build
cmake .. \ -DMMDEPLOY_BUILD_SDK=ON \ -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \ -DCMAKE_CXX_COMPILER=g++-9 \ -Dpplcv_DIR=/usr/local/aarch64-linux-gnu/lib/cmake/ppl \ -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \ -DMMDEPLOY_TARGET_BACKENDS="ort;trt" \ -DMMDEPLOY_CODEBASES=all \
-DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
-DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
-DTENSORRT_DIR=${TENSORRT_DIR} \
-Dpplcv_DIR=/usr/local/aarch64-linux-gnu/lib/cmake/ppl \ -DCUDNN_DIR=${CUDNN_DIR}
cmake --build . -- -j$(nproc) && sudo cmake --install . sudo ldconfig
```

2. PyTorch Model to TensorRT Converting
```
export MODEL_CONFIG=~/detection3d/mmdetection3d/configs/smoke/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d.py
export MODEL_PATH=~/detection3d/mmdetection3d/checkpoints/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth
export TEST_DATA=~/detection3d/mmdetection3d/data/kitti/training/image_2/000001.png

cd ~/mmdeploy
python3 tools/deploy.py configs/mmdet3d/smoke-detection/smoke-detection_tensorrt_static.py $MODEL_CONFIG $MODEL_PATH $TEST_DATA --work-dir smoke_trt --device cuda:0
```