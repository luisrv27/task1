import torch as pt
from itertools import combinations_with_replacement

def generate_polynomial_features(X: pt.Tensor, M: int) -> pt.Tensor:
    N, D = X.size()
    terms = []  
    terms.extend(pt.prod(X[:, list(comb)], dim=1, keepdim=True) for m in range(M + 1) for comb in combinations_with_replacement(range(D), m))
    return pt.cat(terms, dim=1)


def logistic_fun(w: pt.Tensor, M: int, x: pt.Tensor) -> pt.Tensor:
    return pt.sigmoid(generate_polynomial_features(x, M) @ w)

import torch as pt

def binomial_coeff(n: int, k: int) -> int:
    if k > n or k < 0:
        return 0
    num = pt.arange(n - k + 1, n + 1, dtype=pt.int64).prod()  
    denom = pt.arange(1, k + 1, dtype=pt.int64).prod()
    return (num // denom).item()  


def calculate_weights(p: int) -> pt.Tensor:
    z = pt.arange(p, dtype=pt.float32)  
    w = (-1) ** z * (z + 1).sqrt() / (z + 1)
    return w

