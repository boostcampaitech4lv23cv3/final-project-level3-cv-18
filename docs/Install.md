# Install

HW : CUDA Toolkit 11.7.1를 지원하는 HW가 필요합니다.

requirements : python 3.8.5, pipenv, git

Clone git
```bash
git clone https://github.com/boostcampaitech4lv23cv3/final-project-level3-cv-18.git
```

Install library
```bash
cd ~/final-project-level3-cv-18
pipenv install --dev
pipenv shell
```

Install Packages, CUDA
```bash
apt update && \
apt-get install -y --no-install-recommends apt-utils && \
apt-get dist-upgrade -y --no-install-recommends && \
DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata && \
apt-get install -y build-essential cmake pkg-config autoconf software-properties-common \
	gcc g++ gfortran zlib1g-dev cpio curl libboost-dev sudo xz-utils xauth x11-apps \
	libgl1-mesa-glx kmod dkms zip unzip git vim p7zip curl wget htop bash-completion && \
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
sudo sh cuda_11.7.1_515.65.01_linux.run
```

Add path
```bash
ldconfig
cat <<EOF >> /etc/bash.bashrc
export CUDA_TOOLKIT_PATH="/usr/local/cuda"
export CUDNN_INSTALL_PATH="/usr/local/cuda"
export CUDA_HOME="/usr/local/cuda"
export C_INCLUDE_PATH="\${CUDA_HOME}/include:\${C_INCLUDE_PATH}"
export CPATH="\${CUDA_HOME}/include:\${CPATH}"
export PATH="\${CUDA_HOME}/bin:\$PATH"
EOF
```

Install CuDNN
```bash
tar Jxvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
mv cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib/* /usr/local/cuda/lib64/
mv cudnn-linux-x86_64-8.6.0.163_cuda11-archive/include/* /usr/local/cuda/include/
rm /usr/lib/x86_64-linux-gnu/libcudnn*
ldconfig
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt install libnccl2=2.14.3-1+cuda11.7 libnccl-dev=2.14.3-1+cuda11.7
mkdir -p /usr/local/nccl/lib && mkdir -p /usr/local/nccl/include && \
mv /usr/lib/x86_64-linux-gnu/libnccl* /usr/local/nccl/lib/ && \
mv /usr/include/nccl* /usr/local/nccl/include/ && \
echo /usr/local/nccl/lib > /etc/ld.so.conf.d/nccl.conf
ldconfig
```

Install TensorRT
```bash
export trt_depend_version="8.5.3-1+cuda11.8"
export trt_version="8.5.3.1-1+cuda11.8"
dpkg -i /tmp/nv-tensorrt-local-repo-ubuntu1804-8.5.3-cuda-11.8_1.0-1_amd64.deb && \
apt-key add /var/nv-tensorrt-local-repo-ubuntu1804-8.5.3-cuda-11.8/*.pub && \
apt-get update && \
apt-get install -y  libnvinfer8=${trt_depend_version} libnvinfer-plugin8=${trt_depend_version} libnvparsers8=${trt_depend_version} libnvonnxparsers8=${trt_depend_version} \
libnvinfer-bin=${trt_depend_version} libnvinfer-dev=${trt_depend_version} libnvinfer-plugin-dev=${trt_depend_version} libnvparsers-dev=${trt_depend_version} \
libnvonnxparsers-dev=${trt_depend_version} libnvinfer-samples=${trt_depend_version} \
tensorrt=${trt_version} tensorrt-dev=${trt_version} tensorrt-libs=${trt_version} && \
apt-get install -y  python3-libnvinfer=${trt_depend_version} graphsurgeon-tf=${trt_depend_version} \
python3-libnvinfer-dev=${trt_depend_version} uff-converter-tf=${trt_depend_version} onnx-graphsurgeon=${trt_depend_version}
tar xvfz TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-10.2.cudnn8.6.tar.gz
pip install TensorRT-8.5.3.1/python/tensorrt-8.5.3.1-cp38-none-linux_x86_64.whl
```

Install Torch
```bash
pip3 install -U "torch<1.14.0,>=1.13.0" torchvision torchaudio "numpy==1.23.5" scipy
```

Install Tensorrt
```bash
pip3 install torch-tensorrt==1.3.0 --find-links https://github.com/pytorch/TensorRT/releases/expanded_assets/v1.3.0
```

Install ONNX Runtime-GPU
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.13.1/onnxruntime-linux-x64-tensorrt-1.13.1.tgz
tar xvfz onnxruntime-linux-x64-tensorrt-1.13.1.tgz
mkdir -p /opt/onnxruntime
mv onnxruntime-linux-x64-tensorrt-1.13.1/lib /opt/onnxruntime/
mv onnxruntime-linux-x64-tensorrt-1.13.1/include /opt/onnxruntime/
echo "/opt/onnxruntime/lib" > /etc/ld.so.conf.d/onnxruntime.conf
ldconfig
pip install "onnx==1.13.0" "onnxruntime-gpu==1.13.1"
```

Install MMDetection3D
```bash
pip install -U openmim
mim install 'mmengine==0.4.0'
mim install 'mmcv==2.0.0rc3'
mim install 'mmdet==3.0.0rc5'
cd ~/detection3d/mmdetection3d
pip install --ignore-installed PyYAML
pip install -e .
```

Install OpenCV
```bash
apt-get update && \
apt-get install -y ffmpeg libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libgl1-mesa-dri \
libjpeg-dev libpng-dev libpng++-dev libtiff5-dev libdc1394-22-dev opencl-headers mesa-utils libgtkgl2.0-dev \
qtbase5-dev libxine2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgtk2.0-dev libgtkglext1-dev \
libgstreamer-plugins-good1.0-dev libavfilter-dev libx11-dev libglu1-mesa-dev libx264-dev libva-dev \
libavresample-dev libeigen3-dev libv4l-dev v4l-utils libavfilter-dev libxvidcore-dev libcurl4-openssl-dev \
freeglut3-dev libgoogle-glog-dev protobuf-compiler libprotobuf-dev libhdf5-dev libgflags-dev libgphoto2-dev && \
export OPENCV_VERSION=4.5.5
cd /tmp
git clone --depth 1 -b ${OPENCV_VERSION} https://github.com/Itseez/opencv_contrib.git && \
git clone --depth 1 -b ${OPENCV_VERSION} https://github.com/Itseez/opencv.git
mkdir opencv/build && cd opencv/build && \
	cmake -D CMAKE_BUILD_TYPE=Release \
		  -D CMAKE_INSTALL_PREFIX=/usr/local/opencv \
		  -D ENABLE_FAST_MATH=ON \
		  -D ENABLE_PRECOMPILED_HEADERS=ON \
		  -D BUILD_EXAMPLES=OFF \
		  -D BUILD_ANDROID_EXAMPLES=OFF \
		  -D BUILD_PERF_TESTS=OFF \
		  -D BUILD_DOCS=OFF \
		  -D BUILD_TESTS=OFF \
		  -D BUILD_opencv_dnn=ON \
		  -D WITH_FREETYPE=ON \
		  -D CUDA_ARCH_BIN="7.5" \
		  -D CUDA_ARCH_PTX="7.5" \
		  -D CUDA_FAST_MATH=ON \
		  -D WITH_CUDA=ON \
		  -D WITH_CUBLAS=ON \
		  -D WITH_NVCUVID=ON \
		  -D WITH_OPENGL=ON \
		  -D WITH_TBB=ON \
		  -D WITH_QT=ON \
		  -D WITH_IPP_A=ON \
		  -D WITH_XINE=ON \
		  -D WITH_NGRAPH=ON \
		  -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
		  -D BUILD_opencv_python3=ON \
		  -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
		  -D PYTHON3_EXECUTABLE=$(which python3) \
 		  -D PYTHON3_PACKAGES_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])") \
		  -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy, os; print(os.path.join(numpy.__path__[0], 'core/include'))") \
		  -D PYTHON3_INCLUDE_DIR=$(python3 -c "from sysconfig import get_path; print(get_path('include'))") \
		  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ .. && \
	make -j"$(nproc)" && make install && \
	echo "/usr/local/opencv/lib" | tee -a /etc/ld.so.conf.d/opencv.conf && ldconfig
```

Install MMDeploy
```bash
export ONNXRUNTIME_DIR=/opt/onnxruntime
export TENSORRT_DIR=/usr/include/x86_64-linux-gnu
export CUDNN_DIR=/usr/local/cuda
export INSTALL_PREFIX=/opt/mmdeploy

add-apt-repository ppa:ubuntu-toolchain-r/test
apt update
apt install gcc-9 g++-9

update-alternatives --remove-all cuda
update-alternatives --remove-all cuda-11
rm -rf /usr/local/cuda-11.8
cd /usr/local
ln -s cuda-11.7 cuda

cd /tmp
wget https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-linux-x86_64.sh
chmod a+x cmake-3.25.2-linux-x86_64.sh
mkdir -p /opt/cmake
./cmake-3.25.2-linux-x86_64.sh --prefix=/opt/cmake --skip-license
echo "export PATH=/opt/cmake/bin:\$PATH" >> /etc/bash.bashrc
export PATH=/opt/cmake/bin:$PATH
update-alternatives --install /usr/local/bin/cmake cmake /opt/cmake/bin/cmake 1
update-alternatives --install /usr/local/bin/ccmake ccmake /opt/cmake/bin/ccmake 1

cd /opt
git clone https://github.com/openppl-public/ppl.cv.git
cd ppl.cv/
git checkout v0.7.1
./build.sh cuda

cd /opt/ml/detection3d/mmdeploy/third_party
git clone https://github.com/gabime/spdlog.git
cd spdlog/
git checkout v1.11.0
mkdir -p build && cd build
cmake .. && make -j"$(nproc)"

cd /opt/ml/detection3d/mmdeploy/third_party
git clone https://github.com/pybind/pybind11.git
cd pybind11/
git checkout v2.10.3

cd ~/detection3d/mmdeploy/csrc/mmdeploy/execution
wget https://raw.githubusercontent.com/open-mmlab/mmdeploy/master/csrc/mmdeploy/execution/run_loop.h
cd ~/detection3d/mmdeploy
mkdir build -p && cd build
cmake .. \
  -DMMDEPLOY_BUILD_SDK=ON \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
  -DCMAKE_CXX_COMPILER=g++-9 \
  -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
  -DMMDEPLOY_TARGET_BACKENDS="ort;trt" \
  -DMMDEPLOY_CODEBASES=all \
  -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
  -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
  -DTENSORRT_DIR=${TENSORRT_DIR} \
  -Dpplcv_DIR=/opt/ppl.cv/cuda-build/install/lib/cmake/ppl \
  -DCUDNN_DIR=${CUDNN_DIR}
cmake --build . -- -j$(nproc) && sudo cmake --install .
echo "/opt/mmdeploy/lib" | tee -a /etc/ld.so.conf.d/mmdeploy.conf && ldconfig
cd ~/detection3d/mmdeploy
cat <<EOF >> requirements/runtime.txt
aenum==3.1.11
pycuda
EOF
pip install -e .
```
