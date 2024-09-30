

# Fitting Multilevel Factor Models

joint work with T. Hastie and S. Boyd.
This repository accompanies the [manuscript](https://arxiv.org/abs/2409.12067).

We examine a special case of the multilevel factor model, 
with covariance given by 
[multilevel low rank (MLR) matrix](https://arxiv.org/abs/2310.19214). 
We develop a novel, fast implementation of the expectation-maximization
(EM) algorithm, tailored for multilevel factor models, to maximize the 
likelihood of the observed data.
This method accommodates any hierarchical 
structure and maintains linear time and storage complexities per 
iteration. 
This is achieved through a new efficient technique for computing the inverse 
of the positive definite MLR matrix.
We show that the inverse of an invertible PSD MLR matrix is also an 
MLR matrix with the same sparsity in factors, 
and we use the recursive Sherman-Morrison-Woodbury 
matrix identity to obtain the factors of the inverse.
Additionally, we present an algorithm that computes the Cholesky factorization of 
an expanded matrix with linear time and space complexities, 
yielding the covariance matrix as its Schur complement.
This paper is accompanied by an open-source package that implements the
proposed methods.


In this repository, we provide `mfmodel` package implementing proposed methods.


## Installation
To install `mfmodel` 1) activate virtual environment, 2) clone the repo, 3) from inside the directory run 
```python3
pip install -e .
```
Requirements
* python == 3.9
* [mlrfit](https://github.com/cvxgrp/mlr_fitting) == 0.0.1
* numpy >= 1.21
* scipy >= 1.10
* scikit-learn == 1.1.3
* cvxpy == 1.4.2
* matplotlib == 3.7.1
* numba == 0.55.0



## `hello_world`
See the [`examples/hello_world.ipynb`](https://github.com/cvxgrp/multilevel_factor_model/tree/main/examples/hello_world.ipynb) notebook or explanation below.


**Step 1.** Load the feature vector `Y`, rank allocation `ranks`, and hierarchical paritioning `hpart`.
```python3
import mfmodel as mfm
Y, ranks, hpart = ...
```

**Step 2.** Fit the MFM to the data, create `MFModel` object instance.
```python3
fitted_mfm, _ = mfm.fit(Y, ranks, hpart)
```

Once the $\hat \Sigma$ model has been fitted, we can use it for fast linear algebra:
1. Matrix-vector multiplication $\hat \Sigma x$ by calling
```python3
b = fitted_mfm.matvec(x)
```
2. Linear system solve $\hat \Sigma x = b$
```python3
x = fitted_mfm.solve(v)
``` 
3. Diagonal of $\hat \Sigma^{-1}$
```python3
d = fitted_mfm.diag_inv()
```

See [`examples/inverse_large.ipynb`](https://github.com/cvxgrp/multilevel_factor_model/blob/main/examples/inverse_large.ipynb), where we invert $10^5 \times 10^5$ MLR matrix.



## Example notebooks
See the notebooks in [`examples/`](https://github.com/cvxgrp/multilevel_factor_model/tree/main/examples) folder
that show how to use `mfmodel`.