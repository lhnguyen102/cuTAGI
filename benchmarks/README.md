# Results For UCI Small and Large Regression Tasks

## Instructions to set-up your environment
1. Once a virtual environment is created
using:
```
conda create --name your_env_name python=3.10
```
2. install additional modules using:
```
pip install -r benchmarks/requirements.txt
```
3. if not already done, build pytagi using:
```
pip install .
```

## Benchmark Directory Structure
```
.
├── benchmarks                                              # main folder
│   ├──data                                                 # data folder
|   │   ├──UCI                                              # contains the UCI datasets
│   ├──logs                                                 # contains the results
│   ├──README.md                                            # contains instructions
│   ├──requirements.txt                                     # contains the modules
│   ├──small_uci_regression_earlystop.py                    # TAGI early stop script
│   ├──small_uci_regression_optimal_epoch.py                # TAGI optimal epoch script
│   ├──small_uci_regression.py                              # TAGI user-defined epochs script
│   ├──small_uci_regression_het_earlystop.py                # TAGI-V early stop script
│   ├──small_uci_regression_het_optimal_epoch.py            # TAGI-V optimal epoch script
│   ├──small_uci_regression_het.py                          # TAGI-V user-defined epochs script
│   ├──small_uci_regression_het_gain_1_earlystop.py         # TAGI-V early stop script with gain=1
│   ├──small_uci_regression_het_gain_1_optimal_epoch.py     # TAGI-V optimal epoch script with gain=1
│   ├──small_uci_regression_het_gain_1.py                   # TAGI-V user-defined epochs script with gain=1
```

## Small UCI Regression Tasks

### Introduction
In this section, we compare three different setups for running the small UCI regression datasets, namely `Boston_housing`, `Concrete`, `Energy`, `Wine`, `Yacht`, `Kin8nm`, `Naval`, `Power`, and `Protien`.
First, running the benchmark datasets with homoscedastic TAGI for which the error standard deviations for each dataset are obtained by grid-search. Second, running with heteroscedastic error variance inferred using AGVI (Approximate Gaussian Variance Inference) but with modified gain parameters. This setup is referred to as TAGI-V. Here, two gain parameters are introduced, namely `OUT_GAIN` and `NOISE_GAIN` where OUT_GAIN value modifies the initial variance associated with all the weights connecting to the output node representing the expected value while the NOISE_GAIN is associated with only the weights connecting the last layer to the error variance output node. The final setup is to run TAGI-V with both gain parameters set to 1. This is the hyper-parameter free version whereas the modified gain version requires grid-searching for both the parameters. Please refer to the [paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231223013061?via%3Dihub) for more details.

### Implementation Setup
Each dataset is randomly split into a training and a test set having 90% and 10% of the data, respectively, and we maintain the same indices in both sets for each method. We consider 20 data splits to compute the average test performance. For comparative purposes, we consider a network having a single hidden layer of 50 units for each dataset, except for Protein, which has 100 units. The data is normalized, a ReLU activation function is used, and the batch size considered is 10 except for yacht which is 5. An early-stopping procedure is used to identify the optimal number of epochs for each dataset by dividing the training set into an 90 − 10% train-validation sets. The patience is set to 5 and a delta equal to 0.01.

### Homoscedastic TAGI
 To run TAGI with homoscedastic error variance, there are three files namely `small_uci_regression_earlystop.py`, `small_uci_regression_optimal_epoch.py`, and `small_uci_regression.py`. The `small_uci_regression_earlystop.py` script provides the optimal epoch for all 20 splits for each dataset, `small_uci_regression_optimal_epoch.py` provides the result for running until the optimal epoch. These results are averaged over 5 different random seeds. And the last script `small_uci_regression.py` allows running for an user-defined number of epochs and visualizing the learning curves. The results for each of these scripts are stored within the `benchmarks/logs` folder with the same naming convention.

#### Hyper-Parameters
The hyper-parameters in this setup are the homoscedastic error variances. These are obtained by grid-searching over values from 0.1 to 0.8. Here are the values corresponding to each dataset:
```
sigma_v_values = {"Boston_housing": 0.3, "Concrete": 0.3, "Energy": 0.1, "Yacht": 0.1, "Wine": 0.7,
                        "Kin8nm": 0.3, "Naval": 0.6, "Power-plant": 0.2, "Protein": 0.7}
```

#### Commands to run each python script

1. Obtain the optimal epoch for all 20 splits by running the following command. The results will be stored in the folder `benchmarks/logs/results_small_uci_regression_earlystop` such that it contains the test log-likelihood, optimal epoch, test RMSE, and training time with early-stopping.
```
python -m benchmarks.small_uci_regression_earlystop
```

2. You can then run the datasets with their optimal epochs using the following command. Note that the results are averaged over 5 random seeds.

```
python -m benchmarks.small_uci_regression_optimal_epoch
```

3. Or, you can simply run this command to run the datasets for an user-defined number of epochs and visualize the learning curves. For example, the learning curve for Boston_housing with the baseline version can be found here -> `benchmarks/logs/results_small_uci_regression/Boston_housing/RMSE_LL.png`

```
python -m benchmarks.small_uci_regression
```
4. Finally, you can run the following command to produce the benchmark results in a tabular format. The results are stored in `benchmarks/logs/results_small_uci_regression_optimal_epoch/small_uci_regression_table.txt`

```
python -m benchmarks.benchmark_table_small_uci_regression
```
### TAGI-V with modified gain parameters
Similarly, the python scripts for TAGI-V with modified gain has the `het` in the naming. The scripts are named as `small_uci_regression_het_earlystop.py`, `small_uci_regression_het_optimal_epoch.py` and `small_uci_regression_het.py`.

#### Hyper-parameters
The gain parameters `OUT_GAIN` and `NOISE_GAIN` as described in the Introduction Section are the two hyper-parameters in this setup. The values used in the codes are taken directly from Table D.1 of the Supplementary material of the paper `Analytically tractable heteroscedastic uncertainty quantification in Bayesian neural networks for regression tasks` which are obtained by grid-search. These values are as follows:
```
OUT_GAIN = {"Boston_housing": 0.5, "Concrete": 0.5, "Energy": 0.5, "Yacht": 0.1, "Wine": 0.1,
                        "Kin8nm": 1, "Naval": 0.5, "Power-plant": 0.5, "Protein": 0.5}
NOISE_GAIN = {"Boston_housing": 0.01, "Concrete": 0.01, "Energy": 0.01, "Yacht": 0.1, "Wine": 0.01,
                        "Kin8nm": 0.01, "Naval": 0.01, "Power-plant": 0.001, "Protein": 0.1}
```

#### Commands to run the python scripts
```
python -m benchmarks.small_uci_regression_het_earlystop
```
```
python -m benchmarks.small_uci_regression_het_optimal_epoch
```
```
python -m benchmarks.small_uci_regression_het
```
```
python -m benchmarks.benchmark_table_small_uci_regression_het
```

### TAGI-V with Gain = 1
The scripts are named as `small_uci_regression_het_gain_1_earlystop.py`, `small_uci_regression_het_gain_1_optimal_epoch.py` and `small_uci_regression_het_gain_1.py`. There are no hyper-parameters for this setup.

#### Commands to run the python scripts
```
python -m benchmarks.small_uci_regression_het_gain_1_earlystop
```
```
python -m benchmarks.small_uci_regression_het_gain_1_optimal_epoch
```
```
python -m benchmarks.small_uci_regression_het_gain_1
```
```
python -m benchmarks.benchmark_table_small_uci_regression_het_gain_1
```


### Final Table

The table presents the results for the three different setups; TAGI, TAGI-V, and TAGI-V with Gain=1. The table provides the test RMSE, log-likelihood, and the average training time per split for each of them in each dataset.

| Dataset        | TAGI (RMSE)   | TAGI (Log-Likelihood) | TAGI (Avg Training Time) | TAGI-V (RMSE) | TAGI-V (Log-Likelihood) | TAGI-V (Avg Training Time) | TAGI-V with Gain=1 (RMSE) | TAGI-V with Gain=1 (Log-Likelihood) | TAGI-V with Gain=1 (Avg Training Time) |
|:---------------|:----------------------|:-------------------------|:-------------------------|:--------------|:-------------------------|:---------------------------|:---------------------------|:--------------------------------------|:--------------------------------------|
| Boston_housing | 3.146 ± 0.910 | -2.578 ± 0.327        | 0.137 ± 0.003            | 3.120 ± 0.881 | -2.541 ± 0.277           | 0.239 ± 0.006              | 3.326 ± 1.032              | -2.568 ± 0.326                        | 0.109 ± 0.001                        |
| Concrete       | 5.865 ± 0.518 | -3.204 ± 0.109        | 0.279 ± 0.001            | 6.048 ± 0.532 | -3.208 ± 0.126           | 0.367 ± 0.002              | 6.274 ± 0.575              | -3.188 ± 0.209                        | 0.269 ± 0.005                        |
| Energy         | 1.317 ± 0.221 | -1.808 ± 0.284        | 0.494 ± 0.005            | 1.882 ± 0.296 | -1.851 ± 0.218           | 0.615 ± 0.003              | 2.243 ± 0.319              | -1.664 ± 0.444                        | 0.265 ± 0.004                        |
| Yacht          | 1.037 ± 0.280 | -1.597 ± 0.115        | 0.140 ± 0.001            | 1.797 ± 0.626 | -1.643 ± 0.567           | 0.608 ± 0.003              | 1.498 ± 0.672              | -1.422 ± 0.703                        | 0.206 ± 0.002                        |
| Wine           | 0.639 ± 0.035 | -0.975 ± 0.066        | 0.214 ± 0.003            | 0.639 ± 0.036 | -0.972 ± 0.063           | 0.296 ± 0.001              | 0.638 ± 0.037              | -0.984 ± 0.105                        | 0.257 ± 0.006                        |
| Kin8nm         | 0.095 ± 0.004 | 0.893 ± 0.060         | 3.964 ± 0.016            | 0.101 ± 0.004 | 0.883 ± 0.037            | 3.479 ± 0.014              | 0.122 ± 0.010              | 0.942 ± 0.041                         | 3.002 ± 0.049                        |
| Naval          | 0.007 ± 0.000 | 3.540 ± 0.037         | 6.336 ± 0.027            | 0.006 ± 0.000 | 3.748 ± 0.075            | 9.838 ± 0.056              | 0.007 ± 0.000              | 3.738 ± 0.066                         | 8.885 ± 0.078                        |
| Power-plant    | 4.129 ± 0.153 | -2.879 ± 0.054        | 1.423 ± 0.006            | 4.148 ± 0.147 | -2.841 ± 0.035           | 1.630 ± 0.019              | 4.163 ± 0.150              | -2.830 ± 0.039                        | 1.639 ± 0.008                        |
| Protein        | 4.615 ± 0.024 | -2.954 ± 0.006        | 9.970 ± 0.045            | 4.644 ± 0.023 | -2.924 ± 0.010           | 15.261 ± 0.049             | 4.726 ± 0.034              | -2.889 ± 0.029                        | 12.622 ± 0.130                       |
