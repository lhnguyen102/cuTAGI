# cuTAGI
cuTAGI is an open source Bayesian Neural Networks based on Tractable Gaussian Approximate Inference (TAGI) theory. Currently, cuTAGI include different types of layers for neural networks such as Full-connected, convolutional, and transpose convolutional layers. cuTAGI performs different tasks such as supervised-learning (e.g. classification and regression) and unsupervised-learning (e.g. autoencoder). 

## Directory folder
.
├── bin                         # Object files
├── cfg                         # User input
├── data                        # Data base
├── include                     # Header file
├── saved_param                 # Saved network's parameter (.csv)
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
