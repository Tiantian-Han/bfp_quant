# Neural Network Training with MX-Compatible Formats

This repository includes Jupyter notebooks showcasing the integration of the MX PyTorch Emulation Library for quantization of ML Models

## Overview of Notebooks

Each notebook in this repository demonstrates different aspects of training and evaluating neural networks with MX-compatible formats. Below is a brief overview:

### detection.ipynb (Main Code)

- Implements a `SimpleCNN` for CIFAR-10 classification using standard FP32 and dynamic precision switching between FP32, FP4, and FP8 based on runtime performance.
- Includes advanced model management features for precision switching during runtime anomalies.

### baseline.ipynb

- Provides a baseline model using PyTorch's sequential API for MNIST classification.
- Simple architecture and training loops for performance comparison.

### FP32.ipynb

- Trains a `SimpleCNN` model strictly using FP32 precision to serve as a performance benchmark.

### MXFP_CUDA.ipynb

- Integrates the MX PyTorch Emulation Library to train `SimpleCNN` using MX formats.
- Demonstrates custom CUDA extensions for performance optimization.

### Tensor_quant.py

- Utility notebook for tensor quantization using the MX library.
- Demonstrates the process of converting tensors to MX-compatible and bfloat formats.


## Getting Started

To set up the repository for running these notebooks:

1. Follow instructions on installing the [Microxcaling library](https://github.com/microsoft/microxcaling)
2. Install Python dependencies listed in `requirements.txt`.
