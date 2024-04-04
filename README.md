# Table of Contents
* [What is cuTAGI ?](#What-is-cuTAGI)
* [Python Installation](#pytagi-installation)
* [C++/CUDA Installation](#cutagi-installation)
* [Directory Structure](#directory-structure)
* [License](#license)
* [Related Papers](#related-papers)
* [Citation](#citation)

## What is cuTAGI ?
cuTAGI is an open-source Bayesian neural networks library that is based on Tractable Approximate Gaussian Inference (TAGI) theory. cuTAGI includes several of the common neural network layer architectures such as full-connected, convolutional, and transpose convolutional layers, as well as skip connections, pooling and normalization layers. cuTAGI is capable of performing different tasks such as supervised learning, unsupervised learning, and reinforcement learning. The library includes some of the advanced features such as the capacity to propagate uncertainties from the input to the output layer using the the [full covariance mode for hidden layers](https://www.jmlr.org/papers/volume22/20-1009/20-1009.pdf), the capacity to estimate the [derivative](https://www.jmlr.org/papers/volume23/21-0758/21-0758.pdf) of a neural network, and the capacity to quantify heteroscedastic aleatory uncertainty.

cuTAGI is under development and new features will be added as they are ready. Currently supported tasks are:
* Supervised learning
  * Regression
  * Long Short-Term Memory (LSTM)
  * Classification using fully-connected, convolutional and residual architectures
* Unsupervised learning
  * Autoencoders

Coming soon...
* Unsupervised learning: GANs
* Reinforcement learning: DQN
* +++

## Example
```Python
from pytagi.nn import Linear, OutputUpdater, ReLU, Sequential
import pytagi.metric as metric
from pytagi import Utils
from python_examples.data_loader import MnistDataloader

# Load data
x_train_file = "data/mnist/train-images-idx3-ubyte"
y_train_file = "data/mnist/train-labels-idx1-ubyte"

dtl = MnistDataloader(batch_size=batch_size)
dataset = dtl.process_data(x_train_file, y_train_file, x_test_file, y_test_file)
(x_train, y_train, y_train_idx, label_train) = dataset["train"]

# Hierachical Softmax
utils = Utils()
hr_softmax = utils.get_hierarchical_softmax(10)


model = Sequential(Linear(784, 100), ReLU(), Linear(100, 100), ReLU(), Linear(100, 11))
output_updater = OutputUpdater(model.device)

var_obs = np.zeros((batch_size, hr_softmax.num_obs)) + sigma_v**2
batch_iter = dtl.train_batch_generator(x_train, y_train, y_train_idx, label_train, batch_size)
for x, y, y_idx, label in batch_iter:
  # Feed forward
  model(x)

  # Update output layers based on targets
  output_updater.update_using_indices(model.output_z_buffer, y, var_obs, y_idx, model.input_delta_z_buffer)

  # Update parameters
  model.backward()
  model.step()

  # Training metric
  m_pred, v_pred = model.get_outputs()
  pred, _ = utils.get_labels(m_pred, v_pred, hr_softmax, 10, batch_size)
  error_rate = metric.classification_error(prediction=pred, label=label)

```
## License

cuTAGI is released under the MIT license.

**THIS IS AN OPEN SOURCE SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT. NO WARRANTY EXPRESSED OR IMPLIED.**
## Related Papers

* [Tractable approximate Gaussian inference for Bayesian neural networks](https://www.jmlr.org/papers/volume22/20-1009/20-1009.pdf) (James-A. Goulet, Luong-Ha Nguyen, and Said Amiri. JMLR, 2021, 20-1009, Volume 22, Number 251, pp. 1-23.)
* [Analytically tractable hidden-states inference in Bayesian neural networks](https://www.jmlr.org/papers/volume23/21-0758/21-0758.pdf) (Luong-Ha Nguyen and James-A. Goulet. JMLR, 2022, 21-0758, Volume 23, pp. 1-33.)
* [Analytically tractable inference in deep neural networks](https://arxiv.org/pdf/2103.05461.pdf) (Luong-Ha Nguyen and James-A. Goulet. 2021, Arxiv:2103.05461)
* [Analytically tractable Bayesian deep Q-Learning](https://arxiv.org/pdf/2106.11086.pdf) (Luong-Ha Nguyen and James-A. Goulet. 2021, Arxiv:2106.1108)
* [Analytically tractable heteroscedastic uncertainty quantification in Bayesian neural networks for regression tasks](http://profs.polymtl.ca/jagoulet/Site/Papers/Deka_TAGIV_2024_preprint.pdf) (Bhargob Deka, Luong-Ha Nguyen and James-A. Goulet. Neurocomputing, 2024)
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
