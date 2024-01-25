# conda-cuda
Guide to install cuda and cudnn with Anaconda on Ubuntu 22.04

Versions:

+ cuda 11.8
+ cudnn 8.8
+ cudatoolkit 11.8
+ tensorflow 2.13
+ tensorrt 8.5.3.1

[tensorflow-conda-nvidia-gpu-on-ubuntu-22-04-3-lts](https://medium.com/@dev-charodeyka/tensorflow-conda-nvidia-gpu-on-ubuntu-22-04-3-lts-ad61c1d9ee32)


```
conda create -n tf213 python=3.10

conda activate tf213

conda install -c conda-forge cudatoolkit=11.8 cudnn=8.8
```


```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

conda deactivate
```


```
conda activate tf213

echo $LD_LIBRARY_PATH

python -m pip install tensorrt==8.5.3.1

TENSORRT_PATH=$(dirname $(python -c "import tensorrt;print(tensorrt.__file__)"))
echo $TENSORRT_PATH

#linking tensorrt library files to LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/parlange/anaconda3/envs/tf213/lib/python3.10/site-packages/tensorrt' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

conda deactivate
```

```
conda activate tf213

echo $LD_LIBRARY_PATH

python -m pip install tensorflow==2.13

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

```


[Tensorflow with GPU on Linux or WSL2](https://medium.com/@mass.thanapol/tensorflow-with-gpu-on-linux-or-wsl2-10b02fd19924)


```
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## In Ubuntu 22.04 you may get this error
> Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice.
...
Couldn't invoke ptxas --version
...
InternalError: libdevice not found at ./libdevice.10.bc [Op:__some_op]


```
# Install NVCC
conda install -c nvidia cuda-nvcc=11.3.58

# Configure the XLA cuda directory
printf 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Copy libdevice file to the required path
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice

cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/
```




