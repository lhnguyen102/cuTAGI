# Results For UCI Small and Large Regression Tasks

## Table of Contents
* [Instructions to set-up your environment](#instructions-to-set-up-your-environment)
* [Benchmark Directory Structure](#benchmark-directory-structure)
* [Small UCI Regression Tasks](#small-uci-regression-tasks)
* [Large UCI Regression Tasks](#large-uci-regression-tasks)

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
First, running the benchmark datasets with homoscedastic TAGI for which the error standard deviations for each dataset are obtained by grid-search. Second, running with heteroscedastic error variance inferred using [AGVI (Approximate Gaussian Variance Inference)](https://onlinelibrary.wiley.com/doi/10.1002/acs.3667), but with optimized gain parameters. This setup is referred to as TAGI-V. Here, two gain parameters are introduced, namely `OUT_GAIN` and `NOISE_GAIN` where OUT_GAIN value modifies the initial variance associated with all the weights connecting to the output node representing the expected value while the NOISE_GAIN is associated with only the weights connecting the last layer to the error variance output node. The final setup is to run TAGI-V with both gain parameters set to 1. This is the hyperparameter-free version whereas the modified gain version requires grid-searching for both the parameters. Please refer to the [paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231223013061?via%3Dihub) for more details.

### Implementation Setup
Each dataset is randomly split into a training and a test set having 90% and 10% of the data, respectively, and we maintain the same indices in both sets for each method. We consider 20 data splits to compute the average test performance. For comparison purposes, we consider a network having a single hidden layer of 50 units for each dataset, except for Protein, which has 100 units. The data is normalized, a ReLU activation function is used, and the batch size considered is 10 except for yacht which is 5. An early-stopping procedure is used to identify the optimal number of epochs for each dataset by dividing the training set into an 90 − 10% train-validation sets. The patience is set to 5 and a delta equal to 0.01.

### Homoscedastic TAGI
 To run TAGI with homoscedastic error variance, there are three files namely `small_uci_regression_earlystop.py`, `small_uci_regression_optimal_epoch.py`, and `small_uci_regression.py`. The `small_uci_regression_earlystop.py` script provides the optimal epoch for all 20 splits for each dataset, `small_uci_regression_optimal_epoch.py` provides the result for running until the optimal epoch. These results are averaged over 5 different random seeds. And the last script `small_uci_regression.py` allows running for an user-defined number of epochs and visualizing the learning curves. The results for each of these scripts are stored within the `benchmarks/logs` folder with the same naming convention.

#### Hyperparameters
The hyperparameters in this setup are the homoscedastic error variances. These are obtained by grid-searching over values from 0.1 to 0.8. Here are the values corresponding to each dataset:
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
### TAGI-V with optimized gain parameters
Similarly, the python scripts for TAGI-V with optimized gain has the `het` in the naming. The scripts are named as `small_uci_regression_het_earlystop.py`, `small_uci_regression_het_optimal_epoch.py` and `small_uci_regression_het.py`.

#### Hyperparameters
The gain parameters `OUT_GAIN` and `NOISE_GAIN` as described in the Introduction section are the two hyperparameters in this setup. The values used in the codes are taken directly from Table D.1 of the Supplementary material of the paper [Analytically tractable heteroscedastic uncertainty quantification in Bayesian neural networks for regression tasks](https://www.sciencedirect.com/science/article/pii/S0925231223013061?via%3Dihub), which are obtained by grid-search. These values are as follows:
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

The table presents the results for the three different setups; TAGI, TAGI-V, and TAGI-V with Gain=1. The table provides the test RMSE, log-likelihood, and the average training time per split for each of them in each dataset. The computations are done using CPU.

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


## Large UCI Regression Tasks

### Implementation Setup
In this section, TAGI-V is tested for the large UCI regression datasets: `elevators`,
`keggdirected`, `keggundirected`, `pol`, and `skillcraft`. Each dataset is randomly split into a training and test set having 90% and 10% of the data. The experiment is carried out for 10 splits in order to compute the average test RMSE and normalized test log-likelihood values to compare with other benchmark methods. On all datasets, except skillcraft, a network of five hidden layers is used where the number of hidden units in each layer are: [1000, 1000, 500, 50, 2]. For skillcraft, a smaller network is used such that the structure is: [1000, 500, 50, 2]. A ReLU activation unit is used and the batch size considered is 10. The computations are done with `CUDA` instead of `CPU` as there is a clear edge of using `CUDA` for larger network sizes.


### Data-Preprocessing
First pre-processing was to change the input data's standard deviation when it is either equal to zero or it is smaller than 1e-05. Otherwise, there will be numerricals errors due to division by very small number. For this, the folllowing code is used where `x_std` are the standard deviations of the input features of the training set represented by `x_train`.

```
x_mean, x_std = normalizer.compute_mean_std(x_train)
y_mean, y_std = normalizer.compute_mean_std(y_train)

idx = x_std == 0
x_std[x_std < 1e-05] = 1
x_std[idx] = 1
```

Second, was to remove the outliers in the skillcraft dataset using the Z-scores values with a threshold of 5 in order to remove only those few outliers that causes numerical errors. The below code was used to achieve this:

```
def remove_outliers(data, threshold=5):
    """
    Removes outliers from the dataset using Z-score.

    Parameters:
        data (numpy.ndarray): The dataset.
        threshold (float): The threshold for the Z-score. Data points with a Z-score higher than this threshold will be considered outliers.

    Returns:
        numpy.ndarray: Cleaned dataset with outliers removed.
    """
    z_scores = np.abs(stats.zscore(data))
    filtered_data = data[(z_scores < threshold).all(axis=1)]
    return filtered_data
```

### Hyperparameters
The `OUT_GAIN` and `NOISE_GAIN` values are taken directly from the Table H.1 of the Supplementary material
of the paper [Analytically tractable heteroscedastic uncertainty quantification in Bayesian neural networks for regression tasks](https://www.sciencedirect.com/science/article/pii/S0925231223013061?via%3Dihub).
```
OUT_GAIN = {"elevators": 0.1, "pol": 0.5, "skillcraft": 0.1, "keggdirected": 0.1,
        "keggundirected": 0.1}
NOISE_GAIN = {"elevators": 0.1, "pol": 0.001, "skillcraft": 0.1, "keggdirected": 0.0001,
        "keggundirected": 0.1}
```

### Commands to run each python scripts

```
python -m benchmarks.large_uci_regression_het_earlystop
```
```
python -m benchmarks.large_uci_regression_het_optimal_epoch
```
```
python -m benchmarks.large_uci_regression_het
```
```
python -m benchmarks.benchmark_table_large_uci_regression_het
```



### Final Table
The results shown below for the test RMSE, Log-likelihood, and the average training time per split (in s.) are averaged over 3 random seeds and the computations are done using `CUDA`. The results with optimized gains are better, except for keggdirected. However, it is to be noted that obtaining the gain values also require  grid-search.

| Dataset        | RMSE (TAGI-V)      | Log-Likelihood (TAGI-V)   | Avg. Training Time/ split (s.) (TAGI-V) | RMSE (TAGI-V with gain=1) | Log-Likelihood (TAGI-V with gain=1)  | Avg. Training Time/ split (s.) (TAGI-V with gain=1) |
|:---------------|:------------------------|:--------------------------|:----------------------------------------|:-------------------------|:------------------------------------|:---------------------------------------------------|
| elevators      | 0.088 ± 0.002      | -0.335 ± 0.024            | 30.161 ± 3.864                           | 0.093 ± 0.001            | -0.464 ± 0.025                      | 65.819 ± 6.936                                      |
| skillcraft     | 0.262 ± 0.013      | -1.055 ± 0.073            | 4.631 ± 0.400                            | 0.279 ± 0.009            | -1.128 ± 0.047                      | 5.726 ± 0.008                                       |
| pol            | 2.833 ± 0.215      | 0.964 ± 0.135             | 625.676 ± 131.296                       | 3.007 ± 0.228            | 0.640 ± 0.117                       | 866.668 ± 5.315                                     |
| keggdirected   | 0.119 ± 0.006      | 1.147 ± 0.161             | 584.465 ± 132.171                       | 0.117 ± 0.005            | 1.366 ± 0.042                       | 990.828 ± 93.170                                    |
| keggundirected | 0.112 ± 0.004      | 1.809 ± 0.157             | 606.755 ± 47.238                        | 0.112 ± 0.004            | 1.325 ± 0.128                       | 794.805 ± 8.542                                     |


### To-Do

- image regression datasets
- there is discrepency in the results from pytagi to the MATLAB version especially for the small UCI datasets indicating that there is a difference in the setups.
- figure out if there is a solution to bypass the gain parameters -- learning them together with same initial gain will make the the prior variance for the error variance much higher than the expected value. This results in the initial epochs to only learn the error intead of the expected value. This is the reason why the gain parameters needs to be reduced for the error variance so that the expected value can also be learnt in the initial epochs. This is a conundrum: what is the best way to learn both together as there is unidentifiability between them unless one is learnt after the other, i.e. a two-network setup.