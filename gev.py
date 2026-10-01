import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

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


def nll(theta, x, t):
    if t is None:
        mu, log_s, xi = theta
    else:
        mu0, a, log_s, xi = theta
        mu = mu0 + a * t
    if abs(xi) >= 1:  # vraisemblance pas bornée au dela
        return np.inf
    ll = gev_logpdf(x, mu, np.exp(log_s), xi)
    if not np.all(np.isfinite(ll)):
        return np.inf
    return -ll.sum()


def fit_gev(x, t=None):
    """Ajustement GEV par max de vraisemblance.
    Si t est donné : mu = mu0 + alpha * t (sigma et xi constants)."""
    x = np.asarray(x, dtype=float)

    # départ : moments d'une gumbel + regression linéaire pour la pente
    s0 = np.sqrt(6) * x.std() / np.pi
    if t is None:
        theta0 = [x.mean() - 0.5772 * s0, np.log(s0), -0.1]
    else:
        t = np.asarray(t, dtype=float)
        a, b = np.polyfit(t, x, 1)
        theta0 = [b - 0.5772 * s0, a, np.log(s0), -0.1]

    # nelder-mead : pas de gradient à calculer, et le support dépend des paramètres
    res = minimize(nll, theta0, args=(x, t), method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10})
    if not res.success:
        raise RuntimeError(res.message)

    if t is None:
        mu0, log_s, xi = res.x
        alpha = 0.0
    else:
        mu0, alpha, log_s, xi = res.x
    k = len(res.x)
    return {"mu0": mu0, "alpha": alpha, "sigma": float(np.exp(log_s)), "xi": xi,
            "nll": res.fun, "k": k, "aic": 2 * res.fun + 2 * k}


def mu(fit, t=0.0):
    return fit["mu0"] + fit["alpha"] * np.asarray(t, dtype=float)


def borne_sup(fit, t=0.0):
    if fit["xi"] >= 0:
        return np.full(np.shape(t), np.inf)
    return mu(fit, t) - fit["sigma"] / fit["xi"]


def return_level(fit, T, t=0.0):
    T = np.asarray(T, dtype=float)
    return gev_ppf(1 - 1 / T, mu(fit, t), fit["sigma"], fit["xi"])


def return_period(fit, val, t=0.0):
    # infini si val est au dessus de la borne sup
    p = 1 - gev_cdf(val, mu(fit, t), fit["sigma"], fit["xi"])
    with np.errstate(divide="ignore"):
        return 1 / p


def lrt(fit0, fit1):
    D = 2 * (fit0["nll"] - fit1["nll"])
    return D, chi2.sf(D, fit1["k"] - fit0["k"])


def bootstrap(fit, t, stat, n=1000, seed=0):
    """Bootstrap paramétrique : on simule depuis le modèle ajusté (mêmes t que
    les obs), on réajuste et on applique stat() à chaque ajustement."""
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=float)
    out = []
    for i in range(n):
        x2 = gev_sample(mu(fit, t), fit["sigma"], fit["xi"], rng)
        try:
            f = fit_gev(x2, None if fit["k"] == 3 else t)
        except RuntimeError:
            continue
        out.append(stat(f))
    return np.array(out)
