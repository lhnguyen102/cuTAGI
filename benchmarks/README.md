# Benchmarking Results For Regression Tasks

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
├── benchmarks                                  # main folder
│   ├──data                                     # data folder
|   │   ├──UCI                                  # contains the UCI datasets
│   ├──logs                                     # contains the results
│   ├──README.md                                # contains instructions
│   ├──requirements.txt                         # contains the modules
│   ├──small_uci_regression_earlystop.py        # early stop script
│   ├──small_uci_regression_optimal_epoch.py    # optimal epoch script
│   ├──small_uci_regression.py                  # user-defined epochs script
```

## Commands to run each python script

1. Obtain the optimal epoch for all 20 splits by running this command. This will perform the training with early-stopping while considering a validation set. The results will be stored in the folder `benchmarks/logs/results_small_uci_regression_earlystop` such that it contains the test log-likelihood, optimal epoch, test RMSE, and training time with early-stopping.
```
python -m benchmarks.small_uci_regression_earlystop
```

2. You can then run the datasets with their optimal epochs using the following command. Note that the results are averaged over 5 random seeds.

```
python -m benchmarks.small_uci_regression_optimal_epoch
```

3. Or, you can simply run this command to run the datasets for user-defined epochs and see the learning curves. For example, the learning curve for Boston_housing can be found here -> `benchmarks/logs/results_small_uci_regression/Boston_housing/RMSE_LL.png`

```
python -m benchmarks.small_uci_regression
```
4. Finally, you can run the following command to produce the benchmark results. The results are stored in `benchmarks/logs/results_small_uci_regression_optimal_epoch/small_uci_regression_table.txt`

```
python -m benchmarks.benchmark_table_small_uci_regression
```

## Final Tables

### 1. Small UCI Regression with Homoscedastic Error Variance

| Dataset        | RMSE          | Log-Likelihood   | Average Training Time (in s.)   |
|:---------------|:--------------|:-----------------|:--------------------------------|
| Boston_housing | 3.146 ± 0.910 | -2.578 ± 0.327   | 0.137 ± 0.003                   |
| Concrete       | 5.865 ± 0.518 | -3.204 ± 0.109   | 0.279 ± 0.001                   |
| Energy         | 1.317 ± 0.221 | -1.808 ± 0.284   | 0.494 ± 0.005                   |
| Yacht          | 1.037 ± 0.280 | -1.597 ± 0.115   | 0.140 ± 0.001                   |
| Wine           | 0.639 ± 0.035 | -0.975 ± 0.066   | 0.214 ± 0.003                   |
| Kin8nm         | 0.095 ± 0.004 | 0.893 ± 0.060    | 3.964 ± 0.016                   |
| Naval          | 0.007 ± 0.000 | 3.540 ± 0.037    | 6.336 ± 0.027                   |
| Power-plant    | 4.129 ± 0.153 | -2.879 ± 0.054   | 1.423 ± 0.006                   |
| Protein        | 4.615 ± 0.024 | -2.954 ± 0.006   | 9.970 ± 0.045                   |
