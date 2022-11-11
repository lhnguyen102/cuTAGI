# Table of Contents
* [User Input](#user-input)
* [Code Name for Layers and Activation Functions](#code-name-for-layers-and-activation-functions)
* [Network Architecture](#network-architecture)

## User Input
The user inputs are stored as `.txt` that has to be located in the folder `cfg`. User-inputs for cuTAGI are following:
```
model_name:              # Model name, e.g., classification_mnist
task_name:               # Task name, i.e., classification, autoencoder or regression
data_name:               # Data name, e.g., mnist or cifar10
net_name:                # Name of network architecture that is stored in the same folder 
encoder_net_name:        # Name of encoder architecture (This is only for autoencoder task)
decoder_net_name:        # Name of decoder architecture (This is only for autoencoder task)
load_param:              # Load trained model parameters (true or false)
num_epochs:              # Number of epochs
num_classes:             # Number of classes
num_train_data:          # Number of training samples
num_test_data:           # Number of testing samples
mu:                      # Mean of each input, e.g., for 3 channels; mu: 0.5, 0.5, 0.5 
sigma:                   # Standard deviation of each input
data_norm:               # Data normalization (true or false)
x_train_dir:             # Data directory for the training input
y_train_dir:             # Data directory for the training output
x_test_dir:              # Data directory for the testing input
y_test_dir:              # Data directory for the testing output
device:                  # cuda or cpu
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
Fully-connected layer          -> 1
Convolutional layer            -> 2
Tranpose convolutional layer   -> 21
Average pooling layer          -> 4
Layer normalization            -> 5
Batch normalization            -> 6
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


