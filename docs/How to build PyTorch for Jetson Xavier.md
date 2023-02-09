# Xavier 환경 설정
### Base Environment
**Install prerequisites**

```bash
apt install -y --no-install-recommends build-essential software-properties-common libopenblas-dev
apt install -y cmake pkg-config autoconf gcc g++ gfortran clang-9 lld-9 zlib1g-dev cpio curl libboost-dev sudo xz-utils xauth x11-apps
apt install -y protobuf-compiler libprotobuf-dev openssl libssl-dev libcurl4-openssl-dev
apt install -y  bc  gettext-base iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl rsync scons
apt install -y ccache

apt install -y locales locales-all
locale-gen "en_US.UTF-8" && \
update-locale LC_ALL="en_US.UTF-8" LANG="en_US.UTF-8"
```

### Python

```bash
update-alternatives --install /usr/bin/python python /usr/bin/python3 1
apt update
apt install -y python3 python3-dev python3-distutils python3-tk libsnappy-dev
cd tmp
wget https://bootstrap.pypa.io/get-pip.py
python3 /tmp/get-pip.py
pip3 install --upgrade setuptools==59.8.0 wheel nose cython mock pillow
```

**Set Home Directory and Clone Repogitory**

```bash
# Mount NVME
# Git clone
# ...
```

### CMake

```bash
mkdir -p /nvme/opt
wget https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-linux-aarch64.sh
chmod a+x cmake-3.25.2-linux-aarch64.sh
.//nvme/opt.sh
# Install
mv cmake-3.25.2-linux-aarch64 cmake
mv /usr/bin/cmake /usr/bin/_cmake
update-alternatives --install /usr/bin/cmake cmake /nvme/opt/cmake/bin/cmake 1
```

**Dependency**

```bash
pip install ninja pybind11 onnx
```

**scikit-build**

```bash
git clone --branch 0.16.6 https://github.com/scikit-build/scikit-build
cd scikit-build
pip install -e .
```

**pybind11**

```bash
git clone --branch v2.10.3 https://github.com/pybind/pybind11.git
pip install -e .
export pybind11_DIR=/nvme/opt/pybind11/pybind11
```

### ONNX

```bash
export PYVER=3.8
git clone --branch 8.5-GA https://github.com/onnx/onnx-tensorrt.git
cd onnx-tensorrt/third_party/onnx/
export CPLUS_INCLUDE_PATH=/usr/include/python3.8:/usr/local/cuda/targets/aarch64-linux/include
mkdir -p build && cd build
cmake -DCMAKE_CXX_FLAGS=-I/usr/include/python${PYVER} -DBUILD_ONNX_PYTHON=ON -DBUILD_SHARED_LIBS=ON ..
sudo make -j$(nproc) install && \
sudo ldconfig && \
cd .. && \
sudo mkdir -p /usr/include/x86_64-linux-gnu/onnx && \
sudo cp build/onnx/onnx*pb.* /usr/include/x86_64-linux-gnu/onnx && \
sudo cp build/libonnx.so /usr/local/lib && \
sudo rm -f /usr/lib/x86_64-linux-gnu/libonnx_proto.a && \
sudo ldconfig
```

### PyTorch

```bash
git clone -b v1.13.1 --depth=1 --recursive https://github.com/pytorch/pytorch.git
git submodule update --init --recursive --jobs 0
pip install -r requirements.txt

export BUILD_CAFFE2=ON
export BUILD_CAFFE2_OPS=ON
export USE_FBGEMM=OFF
export USE_FAKELOWP=OFF
export BUILD_TEST=OFF
export USE_MKLDNN=OFF
export USE_NNPACK=OFF
export USE_XNNPACK=OFF
export USE_QNNPACK=OFF
export USE_PYTORCH_QNNPACK=OFF
export USE_CUDA=ON
export USE_CUDNN=ON
export TORCH_CUDA_ARCH_LIST="7.2"
export USE_NCCL=OFF
export USE_SYSTEM_NCCL=OFF
export USE_OPENCV=ON
export USE_DISTRIBUTED=ON
export USE_TENSORRT=OFF
export USE_NUMPY=ON
export MAX_JOBS=8
export PATH=/usr/lib/ccache:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc

python3 setup.py clean
python3 setup.py bdist_wheel
```

### Torchvision

```bash
git clone --branch v0.14.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.14.1

python3 setup.py clean
python3 setup.py bdist_wheel
```

**Upgrade cmake**

```bash
cd /opt
export CMAKE_VERSION=3.25.2
wget https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION.tar.gz
tar xpvf cmake-$CMAKE_VERSION.tar.gz
ln -s cmake-$CMAKE_VERSION cmake
cd cmake
./bootstrap --system-curl
make -j8
echo 'export PATH=/opt/cmake/bin/:$PATH' >> /etc/bash.bashrc
source /etc/bash.bashrc
```

**Build ONNXRuntime**

```bash
cd /tmp
git clone --recursive -b v1.13.1 https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --update --build --parallel --build_wheel \
 --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu \
 --tensorrt_home /usr/lib/aarch64-linux-gnu
```

**Build Torchvision**

```bash
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev libopenblas-base libopenmpi-dev  libopenblas-dev -y
git clone --branch v0.14.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.14.1
pip install -e .
```