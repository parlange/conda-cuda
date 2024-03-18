# conda-cuda
Guide to install cuda and cudnn with Anaconda on Ubuntu 22.04

## Versions

+ cuda 11.8
+ cudnn 8.8
+ cudatoolkit 11.8
+ tensorflow 2.13
+ python 3.10
+ tensorrt 8.5.3.1

## Installation

nvidia-driver

```
lspci | grep -e VGA

ubuntu-drivers devices

sudo apt install nvidia-driver-535

```

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

pip install tensorrt==8.5.3.1

TENSORRT_PATH=$(dirname $(python -c "import tensorrt;print(tensorrt.__file__)"))
echo $TENSORRT_PATH

#linking tensorrt library files to LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/parlange/anaconda3/envs/tf213/lib/python3.10/site-packages/tensorrt' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

conda deactivate
```

```
conda activate tf213

echo $LD_LIBRARY_PATH

pip install tensorflow==2.13

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

```


[Tensorflow with GPU on Linux or WSL2](https://medium.com/@mass.thanapol/tensorflow-with-gpu-on-linux-or-wsl2-10b02fd19924)


```
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### In Ubuntu 22.04 you may get this error
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


### Paths for conda env with cuda, cudnn and tensorrt. Edit env_vars.sh file:

```
nano $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Paths copied to env_vars.sh for this conda env:

```
export LD_LIBRARY_PATH=/home/parlange/anaconda3/envs/tf213/lib/
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/parlange/anaconda3/envs/tf213/lib/python3.10/site-packages/tensorrt
```

### Remove NUMA warnings 
https://github.com/tensorflow/tensorflow/issues/42738
```
for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done
```
https://zrruziev.medium.com/fixing-the-successful-numa-node-read-from-sysfs-had-negative-value-1-but-there-must-be-at-7d798d5b912d

```
sudo echo 0 | sudo tee -a /sys/bus/pci/devices/0000\:01\:00.0/numa_node
```


Install jupyter via Anaconda

```
conda install jupyter
```

[Local runtime in Colab](https://research.google.com/colaboratory/local-runtimes.html)

```
  jupyter notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=8888 \
    --NotebookApp.port_retries=0
```

Copy the kernel URL to Local Runtime in Colab to use your GPU with their front end.



## tensorflow-probability with tf autoinstaller

For tensorflow-probability use Tensorflow 2.15

```
conda create -n tf215 python=3.10

conda activate tf215

pip install tensorflow[and-cuda]

pip install tensorflow==2.15

conda install jupyter

conda install -c anaconda ipykernel

python -m ipykernel install --user --name=tf215

```
The auto installer loaded cuDNN version 8906, tensorflow 2.15

```
nvidia-cublas-cu12        12.2.5.6                 pypi_0    pypi
nvidia-cuda-cupti-cu12    12.2.142                 pypi_0    pypi
nvidia-cuda-nvcc-cu12     12.2.140                 pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.2.140                 pypi_0    pypi
nvidia-cuda-runtime-cu12  12.2.140                 pypi_0    pypi
nvidia-cudnn-cu12         8.9.4.25                 pypi_0    pypi
nvidia-cufft-cu12         11.0.8.103               pypi_0    pypi
nvidia-curand-cu12        10.3.3.141               pypi_0    pypi
nvidia-cusolver-cu12      11.5.2.141               pypi_0    pypi
nvidia-cusparse-cu12      12.1.2.141               pypi_0    pypi
nvidia-nccl-cu12          2.16.5                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.2.140                 pypi_0    pypi
tensorboard               2.15.2                   pypi_0    pypi
tensorboard-data-server   0.7.2                    pypi_0    pypi
tensorflow                2.15.0.post1             pypi_0    pypi
tensorflow-estimator      2.15.0                   pypi_0    pypi
tensorflow-io-gcs-fs      0.36.0                   pypi_0    pypi
tensorflow-probability    0.23.0                   pypi_0    pypi
termcolor                 2.4.0                    pypi_0    pypi
```

