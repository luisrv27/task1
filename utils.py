import torch as pt


def generate_polynomial_features(x: pt.Tensor, M: int) -> pt.Tensor:
    N, D = x.shape
    indices = (
        combo for degree in range(1, M + 1)
        for combo in pt.combinations(pt.arange(D), r=degree, with_replacement=True)
    )

    terms = [pt.ones((N, 1), device=x.device)]
    terms.extend(pt.prod(x[:, combo], dim=1, keepdim=True) for combo in indices)

    return pt.cat(terms, dim=1)

def logistic_fun(w: pt.Tensor, M: int, x: pt.Tensor) -> pt.Tensor:
    return pt.sigmoid(generate_polynomial_features(x, M) @ w)

def binomial_coeff(n: int, k: int) -> int:
    if k > n or k < 0:
        return 0
    num = pt.arange(n - k + 1, n + 1, dtype=pt.int64).prod()  
    denom = pt.arange(1, k + 1, dtype=pt.int64).prod()
    return (num // denom).item()  

def calculate_weights(p: int) -> pt.Tensor:
    z = pt.arange(p, dtype=pt.float)
    z_reversed = z.flip(0)
    w = (-1) ** z_reversed * (z_reversed + 1).sqrt() / (z_reversed + 1)
    return w

def generate_data(N:int, M:int, D=5, add_noise=True):
    p  = sum(binomial_coeff(D + m - 1, m) for m in range(M+1))
    w = calculate_weights(p)
    X = pt.empty(N, D).uniform_(-5.0, 5.0)
    y = logistic_fun(w, M, X)
    if add_noise:
        y+= pt.normal(mean=0,std=1, size=y.size())
    targets = (y >= 0.5).int()
    return X, targets