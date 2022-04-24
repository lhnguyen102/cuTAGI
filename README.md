# Table of Contents
* [What is cuTAGI](#What-is-cuTAGI)
* [Installation](#Installation)
* [API](#API)
* [Directory Structure](#directory-structure)
* [Licensing](#licensing)
* [Related Papers](#related-papers)
* [Citation](#citation)

## What is cuTAGI ?
cuTAGI is an open source Bayesian Neural Networks that is based on Tractable Approximate Gaussian Inference (TAGI) theory. Currently, cuTAGI includes different types of layers for neural networks such as Full-connected, convolutional, and transpose convolutional layers. cuTAGI performs different tasks such as supervised-learning (e.g. classification and regression), unsupervised-learning (e.g. autoencoder) and reinforcement learning. 

## Installation
### Ubuntu
To compile all functions, use `make -f Makefile`.

### Window

Comming soon...

NOTE: We currently support Ubuntu 20.04 with a NVIDA GPU and CUDA toolkit >=10.1

## API

Comming soon...

## Directory Structure
```
.
├── bin                         # Object files
├── cfg                         # User input (.txt)
├── data                        # Database
├── include                     # Header file
├── saved_param                 # Saved network's parameters (.csv)
├── saved_results               # Saved network's inference (.csv)
├── src                         # Source files
│   ├── common.cpp              # Common functions 
│   ├── cost.cpp                # Performance metric
│   ├── dataloader.cpp          # Load train and test data
│   ├── data_transfer.cu        # Transfer data host from/to device
│   ├── feed_forward.cu         # Prediction 
│   ├── global_param_update.cu  # Update network's parameters
│   ├── indices.cpp             # Pre-compute indices for network
│   ├── net_init.cpp            # Initialize the network
│   ├── net_prop.cpp            # Network's properties
│   ├── param_feed_backward.cu  # Learn network's parameters
│   ├── state_feed_backward.cu  # Learn network's hidden states
│   ├── task.cu                 # Perform different tasks 
│   ├── user_input.cpp          # User input variable
│   └── utils.cpp               # Different tools
├── config.py                   # Generate network architecture (.txt)
├── main.cpp                    # The ui
```

## Licensing 

cuTAGI is released under the MIT license. 

**THIS IS AN OPEN SOURCE SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT. NO WARRANTY EXPRESSED OR IMPLIED.**
## Related Papers 

* [Tractable approximate Gaussian inference for Bayesian neural networks](https://www.jmlr.org/papers/volume22/20-1009/20-1009.pdf) (James-A. Goulet et al., 2021) 
* [Analytically tractable hidden-states inference in Bayesian neural networks](https://www.jmlr.org/papers/volume22/20-1009/20-1009.pdf) (Luong-Ha Nguyen and James-A. Goulet, 2022)
* [Analytically tractable inference in deep neural networks](https://arxiv.org/pdf/2103.05461.pdf) (Luong-Ha Nguyen and James-A. Goulet, 2021)
* [Analytically Tractable Bayesian Deep Q-Learning](https://arxiv.org/pdf/2106.11086.pdf) (Luong-Ha Nguyen and James-A. Goulet)

## Citation

```
@misc{
  Author = {Luong-Ha Nguyen and James-A. Goulet},
  Title = {Tractable Approximate Gaussian Inference with CUDA programming},
  Year = {2022},
}
```
