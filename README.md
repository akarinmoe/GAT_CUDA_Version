# Curriculum Design for Principles of Parallel Programming@BIT

Curriculum Design for Principles of Parallel Programming Advised by Professor Yizhuo Wang

Student: Weichu Xie 1120222370

## Hardware

GPU: RTX 3090 with CUDA 11.8 sm_86

CPU: Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz

## Introduction

### Contents

- CUDA implement of GraphAttentionLayer
- A script for inference and its time consumption
- Python package of kernels using setup.py

### Main Structure

```
GAT_CUDA/
├── data/                                   # Cora dataset
│   └── cora/
│       ├── cora.cites
│       ├── cora.content
│       └── README
├── kernel_folder/					
│   ├── Kernels/                            # CUDA kernels
│   │   ├── aggregate_features.cu
│   │   ├── compute_attention_coeff.cu
│   │   ├── gat.cpp
│   │   ├── linear_transform.cu
│   │   ├── softmax_kernel.cu
│   │   ├── makefile                        # Makefile of the kernel folder
│   └── build/
├── .gitignore
├── 855.pkl                                 # Checkpoint of core dataset
├── inference.py                            # Inference script
├── layers.py
├── models.py
├── README.md                               # Main project documentation
├── requirements.txt                        # Dependencies required for the project
├── setup.py
├── train.py
└── utils.py
```



## Environment

```
python = 3.8.19
```



## Demo

```
pip install -r requirements.txt

cd kernel_folder/Kernels

make
```

Tips: The `makefile` is used to demonstrate the using of `nvcc` and `g++` to compile the project, but the real project is conducted using `ninja`



## QuickStart

```
python setup.py install

python inference.py --checkpoint 855.pkl
```

If you want to use CPU, please add the args `--no-cuda` in the command `python inference.py --checkpoint 855.pkl`

Here is the GPU result: 

```
Using /home/xwc/.cache/torch_extensions/py38_cu118 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/xwc/.cache/torch_extensions/py38_cu118/gat_cuda/build.ninja...
Building extension module gat_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module gat_cuda...
Loading cora dataset...
Batch 0-200 results: loss= 4.4720, accuracy= 0.3000, time= 0.0037s
Batch 200-400 results: loss= 3.9312, accuracy= 0.3900, time= 0.0001s
Batch 400-600 results: loss= 4.4456, accuracy= 0.2950, time= 0.0001s
Batch 600-800 results: loss= 3.9455, accuracy= 0.3750, time= 0.0001s
Batch 800-1000 results: loss= 4.6790, accuracy= 0.2750, time= 0.0001s

Overall Inference results:
Test loss: 4.4439
Test accuracy: 0.3090
Total time elapsed for inference: 0.0113s
```



## Future Work

- Train process of the GAT (Due to PyTorch's automatic backpropagation mechanism for operators, we can inherit from `nn.Module` to implement the gradient update process of CUDA operators during backpropagation.)

- Performance comparison between custom operators and PyTorch framework CUDA



## Reference

- Slides of Curriculum Principles of Parallel Programming
- [pytorch tutorial](https://pytorch-cn.readthedocs.io/zh/latest/)
- [NVIDIA CUDA Programming Guide](https://www.nvidia.cn/docs/IO/51635/NVIDIA_CUDA_Programming_Guide_1.1_chs.pdf)