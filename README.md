# Table of Contents
* [What is cuTAGI ?](#What-is-cuTAGI)
* [User Input](#user-input)
* [Code Name for Layers and Activation Functions](#code-name-for-layers-and-activation-functions)
* [Network Architecture](#network-architecture)
* [Installation](#Installation)
* [API](#API)
* [Directory Structure](#directory-structure)
* [License](#license)
* [Related Papers](#related-papers)
* [Citation](#citation)

## What is cuTAGI ?
cuTAGI is an open-source Bayesian neural networks library that is based on Tractable Approximate Gaussian Inference (TAGI) theory. cuTAGI includes several of the common neural network layer architectures such as full-connected, convolutional, and transpose convolutional layers, as well as skip connections, pooling and normalization layers. cuTAGI is capable of performing different tasks such as supervised learning, unsupervised learning, and reinforcement learning.

cuTAGI is under development and new features will be added as they are ready. Currently supported tasks are:
* Supervised learning: Regression and classification using fully-connected, convolutional and residual architectures
* Unsupervised learning: autoencoders
 
Coming soon...
* Unsupervised learning: GANs
* Reinforcement learning: DQN
* +++

## User Input
The user inputs are stored as `.txt` that has to be located in the folder `cfg`. User-inputs for cuTAGI are following:
```
model_name:              # Model name, e.g., classification_mnist
task_name:               # Task name, i.e., classification, autoencoder or regression
data_name:               # Data name, e.g., mnist or cifar10
net_name:                # Name of network architecture that is stored in the same folder 
encoder_net_name:        # Name of encoder architecture (This is only for autoencoder task)
decoder_net_name:        # Name of decoder architecture (This is only for autoencoder task)
load_param:              # Do we want to load network's parameters that has been trained
num_epochs:              # Number of epochs
num_classes:             # Number of classes
num_train_data:          # Number of training samples
num_test_data:           # Number of testing samples
mu:                      # Mean of each input, e.g., for 3 channels; mu: 0.5, 0.5, 0.5 
sigma:                   # Standard deviation of each input
x_train_dir:             # Data directory for the training input
y_train_dir:             # Data directory for the training output
x_test_dir:              # Data directory for the testing input
y_test_dir:              # Data directory for the testing output
```
The default values for each input user is set to empty. Here is an example of user inputs for the MNIST classification task [`cfg/cfg_mnist_2conv.txt`](https://github.com/lhnguyen102/cuTAGI/blob/main/cfg/cfg_mnist_2conv.txt)
```
model_name: mnist_2conv
task_name: classification
data_name: mnist
net_name: 2conv
load_param: false
num_epochs: 2
num_classes: 10
num_train_data: 60000
num_test_data: 10000
mu: 0.1309
sigma: 1
x_train_dir: data/mnist/train-images-idx3-ubyte
y_train_dir: data/mnist/train-labels-idx1-ubyte
x_test_dir: data/mnist/t10k-images-idx3-ubyte
y_test_dir: data/mnist/t10k-labels-idx1-ubyte
```
## Code Name for Layers and Activation Functions
Each layer type is assigned to an integer number
```
Full-connected layer          -> 1
Convolutional layer           -> 2
Tranpose convolutional layer  -> 21
Average pooling layer         -> 4
Layer normalization           -> 5
Batch normalization           -> 6
```

Each activation function is assigned to an integer number
```
No activation  -> 0
Tanh           -> 1
Sigmoid        -> 2
ReLU           -> 4
Softplus       -> 5
LeakyReLU      -> 6
```
An example of the use of these code names can be found in [Network Architecture](#network-architecture).
## Network Architecture
The network architecture (`.txt`) is user-specified and stored in the folder `cfg`. A basic network architecture file is as follow
```
layers:           # Type of layers
nodes:            # Number of hidden units
kernels:          # Kernel size 
strides:          # Increment by which each kernel scans the image
widths:           # Width of the images
heights:          # Height of the images 
filters:          # Number of filters 
pads:             # Number of padding around the images
pad_types:        # Type of paddings
activations:      # Type of activation function
batch_size:       # Number of observation per mini-batches
sigma_v:          # Observation noise's standard deviation
```
Here is an example of user inputs for the mnist classification [`cfg/2conv.txt`](https://github.com/lhnguyen102/cuTAGI/blob/main/cfg/2conv.txt)
```
layers:     [2,     2,      4,      2,      4,      1,      1]
nodes:      [784,   0,      0,	    0,      0,      20,     11]
kernels:    [4,     3,      5,      3,      1,      1,      1]
strides:    [1,     2,      1,      2,      0,      0,      0]
widths:     [28,    0,      0,      0,      0,      0,      0]
heights:    [28,    0,      0,      0,      0,      0,      0]
filters:    [1,     16,     16,     32,     32,     1,      1]
pads:       [1,     0,      0,      0,      0,      0,      0]
pad_types:  [1,     0,      0,      0,      0,      0,      0]
activations:[0,     4,      0,      4,      0,      4,      0]
batch_size: 10
sigma_v:    1
```

## Installation
### Ubuntu 20.04
1. Install [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) >=10.1 in `/usr/local/` and add the CUDA location to PATH. For example, adding the following to your `~/.bashrc`
```sh
export PATH="/usr/local/cuda-10.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH"
```
2. Install GCC compiler by entering this line in `Terminal`
```sh
sudo apt install g++
```
3. Install CMake by following [these instructions](https://cmake.org/install/) 

4. Build the project using CMake by the folder `cuTAGI` and  entering these lines in `Terminal`
```
cmake . -B build
cmake --build build --config RelWithDebInfo -j 16
```

### Windows
1. Download and install MS Visual Studio 2019 community and C/C++ by following [these instructions](https://docs.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170)

2. Install [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) >=10.1 and add CUDA location to Environment variables [(see Step 5.3)](https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781)

3. Copy all extenstion files from CUDA to MS Visual Studio. See this [link](https://github.com/mitsuba-renderer/mitsuba2/issues/103#issuecomment-618378963) for further details.
```
COPY FROM C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\visual_studio_integration\MSBuildExtensions 
TO        C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations
```
4. Download and install CMake [Windows x64 Installer](https://cmake.org/download/) and add the install directory (e.g., `C:\Program Files\CMake\bin`) to PATH in [Environment variables](https://docs.microsoft.com/en-us/previous-versions/office/developer/sharepoint-2010/ee537574(v=office.14))

5. Add CMake CUDA compiler to [Environment variables](https://docs.microsoft.com/en-us/previous-versions/office/developer/sharepoint-2010/ee537574(v=office.14)).  
```
variable = CMAKE_CUDA_COMPILER 
value = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin\nvcc.exe
```
6. Build the project using CMake by navigating to the folder `cuTAGI` and  entering these lines in `Command Prompt`
```
cmake . -B build
cmake --build build --config RelWithDebInfo -j 16
```

*NOTE: Users must enter the CUDA version installed on their machine. Here, we illustrate the installation with CUDA version v10.1 (see Step 1 for Ubuntu and 3 & 5 for Windows). 



## API
Here are terminal command lines that excecutes the classification task for MNIST images using
* Two fully connected layer [`cfg/2fc.txt`](https://github.com/lhnguyen102/cuTAGI/blob/main/cfg/2fc.txt) 
```sh
build/main cfg_mnist_2fc.txt
```
or
```sh
"build/main" cfg_mnist_2fc.txt
```

* Two convolutional layers [`cfg/2conv.txt`](https://github.com/lhnguyen102/cuTAGI/blob/main/cfg/2conv.txt).
```sh
build/main cfg_mnist_2conv.txt
```

* Two convolutional layers each followed by a batch-normalization layer [`cfg/2conv_bn.txt`](https://github.com/lhnguyen102/cuTAGI/blob/main/cfg/2conv_bn.txt)
```cpp
build/main cfg_mnist_2conv_bn.txt
```

## Directory Structure
```
.
├── cfg                         # User input (.txt)
├── data                        # Database
├── include                     # Header files
├── saved_param                 # Saved network's parameters (.csv)
├── saved_results               # Saved network's inference (.csv)
├── src                         # Source files
│   ├── common.cpp              # Common functionalities 
│   ├── cost.cpp                # Performance metric
│   ├── dataloader.cpp          # Load train and test data
│   ├── data_transfer.cu        # Transfer data host from/to device
│   ├── feed_forward.cu         # Prediction 
│   ├── global_param_update.cu  # Update network's parameters
│   ├── indices.cpp             # Pre-compute indices for network
│   ├── net_init.cpp            # Initialize the network
│   ├── net_prop.cpp            # Network's properties
│   ├── param_feed_backward.cu  # Learn network's parameters
│   ├── state_feed_backward.cu  # Learn network's hidden states
│   ├── task.cu                 # Perform different tasks 
│   ├── user_input.cpp          # User input variables
│   └── utils.cpp               # Different tools
├── config.py                   # Generate network architecture (.txt)
├── main.cpp                    # The ui

```

## License 

cuTAGI is released under the MIT license. 

**THIS IS AN OPEN SOURCE SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT. NO WARRANTY EXPRESSED OR IMPLIED.**
## Related Papers 

* [Tractable approximate Gaussian inference for Bayesian neural networks](https://www.jmlr.org/papers/volume22/20-1009/20-1009.pdf) (James-A. Goulet, Luong-Ha Nguyen, and Said Amiri. JMLR, 2021, 20-1009, Volume 22, Number 251, pp. 1-23.) 
* [Analytically tractable hidden-states inference in Bayesian neural networks](https://www.jmlr.org/papers/volume23/21-0758/21-0758.pdf) (Luong-Ha Nguyen and James-A. Goulet. JMLR, 2022, 21-0758, Volume 23, pp. 1-33.)
* [Analytically tractable inference in deep neural networks](https://arxiv.org/pdf/2103.05461.pdf) (Luong-Ha Nguyen and James-A. Goulet. 2021, Arxiv:2103.05461)
* [Analytically tractable Bayesian deep Q-Learning](https://arxiv.org/pdf/2106.11086.pdf) (Luong-Ha Nguyen and James-A. Goulet. 2021, Arxiv:2106.1108)

## Citation

```
@misc{cutagi2022,
  Author = {Luong-Ha Nguyen and James-A. Goulet},
  Title = {cu{TAGI}: a {CUDA} library for {B}ayesian neural networks with Tractable Approximate {G}aussian Inference},
  Year = {2022},
  journal = {GitHub repository},
  howpublished = {https://github.com/lhnguyen102/cuTAGI}
}
```
