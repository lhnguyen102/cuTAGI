# Table of Contents
* [What is cuTAGI ?](#What-is-cuTAGI)
* [Example](#pytagi-installation)
* [License](#license)
* [Related Papers](#related-papers)
* [Citation](#citation)

## What is cuTAGI ?
cu/Py TAGI is an open-source Bayesian neural networks library that is based on Tractable Approximate Gaussian Inference (TAGI) theory. It specializes in quantifying uncertainty in neural network parameters, enabling a range of tasks such as supervised, unsupervised, and reinforcement learning.

Supported Layers:
- [x] Linear
- [x] CNNs
- [x] Transpose CNNs
- [x] LSTMs
- [x] Batch Normalization
- [x] Layer Normalization
- [ ] GRU

Model Development Tools:
- [x] Sequential Model Construction
- [ ] Eager Execution (WIP)

Examples of [regression task](#regression-task) using the diagonal (top left) or full (top right) covariance modes for hidden layers, an example of heteroscedastic aleatory uncertainty inferrence (bottom left), and an example for the estimation of the derivative of a function modeled by a neural network (bottom right).
<p align="center">
  <img  align="left", src="./saved_results/pred_diag_toy_example_disp.png" width="340px">&emsp;&emsp;<img src="./saved_results/pred_full_cov_toy_example_disp.png" width="345px">&emsp;&emsp;<img  align="left", src="./saved_results/pred_hete_toy_example_disp.png" width="348px">&emsp;&emsp;<img src="./saved_results/pred_derivative_toy_example_disp.png" width="335px">
</p>


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
  model.forward(x)

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
