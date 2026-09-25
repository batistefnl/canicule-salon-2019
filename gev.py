import numpy as np

# convention : xi < 0 -> queue bornée (Weibull), xi > 0 -> queue lourde
# attention scipy.stats.genextreme utilise c = -xi


def gev_logpdf(x, mu, sigma, xi):
    z = (np.asarray(x, dtype=float) - mu) / sigma
    if abs(xi) < 1e-6:  # gumbel
        return -np.log(sigma) - z - np.exp(-z)
    y = 1 + xi * z
    with np.errstate(divide="ignore", invalid="ignore"):
        res = -np.log(sigma) - (1 + 1 / xi) * np.log(y) - y ** (-1 / xi)
    return np.where(y > 0, res, -np.inf)


def gev_cdf(x, mu, sigma, xi):
    z = (np.asarray(x, dtype=float) - mu) / sigma
    if abs(xi) < 1e-6:
        return np.exp(-np.exp(-z))
    y = np.maximum(1 + xi * z, 0.0)
    with np.errstate(divide="ignore"):
        res = np.exp(-(y ** (-1 / xi)))
    # y = 0 : sous le support si xi > 0, au dessus de la borne si xi < 0
    return np.where(y > 0, res, 0.0 if xi > 0 else 1.0)


def gev_ppf(p, mu, sigma, xi):
    p = np.asarray(p, dtype=float)
    if abs(xi) < 1e-6:
        return mu - sigma * np.log(-np.log(p))
    return mu + sigma / xi * ((-np.log(p)) ** (-xi) - 1)


def gev_sample(mu, sigma, xi, rng):
    mu = np.asarray(mu, dtype=float)
    return gev_ppf(rng.uniform(size=mu.shape), mu, sigma, xi)