#!/bin/bash
conda create -n ensemble python=3.6
conda activate ensemble
module load cuda/11.2

export CUDA_ROOT_DIR=/usr/local/cuda-11.2
export CUDA_SDK_ROOT_DIR=/usr/local/cuda-samples
export GPUNUFFT_ROOT_DIR=~/Projects/gpuNUFFT
export LDFLAGS=-L/usr/local/cuda-11.2/lib64
export CC=/usr/bin/gcc-8
export CXX=/usr/bin/g++-8

cd ~/Projects/gpuNUFFT/CUDA
mkdir -p build
cd build
cmake .. -DGEN_MEX_FILES=OFF
make

cd ~/Projects/optox
pip install -r requirements.txt
mkdir build
cd build
cmake .. -DWITH_PYTHON=ON -DWITH_PYTORCH=ON -DWITH_TENSORFLOW=ON -DWITH_GPUNUFFT=ON -DCUDA_SDK_ROOT_DIR=/usr/local/cuda-samples -DCUDA_ROOT=/usr/local/cuda-11.2
make install

cd ~/Projects/merlin
./install.sh

cd ~/Projects/ensemble/external/medutils
python setup.py bdist_wheel
pip install dist/Medutils-0.1-py3-none-any.whl

cd ~/Projects/ensemble
pip install -r requirements.txt