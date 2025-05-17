# Statistical Inference for Cluster-based Anomaly Detection

[![PyPI version](https://badge.fury.io/py/si-clad.svg)](https://badge.fury.io/py/si-clad)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This package provides a Statistical Inference framework for Cluster-based Anomaly Detection, with controllable FPR. In particular, we propose a valid p-value for testing the anomaly results obtained by DBSCAN algorithm. Basically, the problem is decomposed into multiple tractable sub-problems to enable an efficient test with the highest TPR while controlling the FPR through the divide-and-conquer approach. For more details, please refer to the paper at: https://arxiv.org/abs/2504.18633

This package has the following requirements:

    mpmath
    numpy>=1.23.0
    scipy>=1.4.1

## Installization

You can install this package from PyPI using:

`pip install si-clad`

### Example
```python 
from si_clad import SI_CLAD, generate, DBSCAN_AD

n = 50
d = 10
delta =  0 #no true outliers
X, Sigma, _ = generate(n, d, delta)

minpts =  10
eps = 3
O, _ = DBSCAN_AD(eps, minpts).fit(X)
p_value = SI_CLAD(X, Sigma, minpts, eps, O, j = None) #randomly choose an outlier j for testing

print(p_value)

```
## Reproducibility

Explore our collection of Jupyter notebooks for hands-on demonstrations of the si-clad package in action.

- Example for computing $p$-value for Cluster-based Anomaly Detection
```
>> ex1_compute_pvalue.ipynb
```

- Check the uniformity of the $p$-values of Cluster-based Anomaly Detection under the null hypothesis $H_0$
```
>> ex2_validity_of_pvalue.ipynb
```

