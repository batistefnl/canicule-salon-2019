import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gev import mu

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
GREY = "#52514e"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": GREY,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.8,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "legend.frameon": False,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def gringorten(n):
    p = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
    return 1 / (1 - p)


def plot_series(years, x, fit, t, rl100, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ev = years == 2019
    ax.plot(years[~ev], x[~ev], "o", ms=4, color=GREY, label="Maximum estival observé")
    ax.plot(years[ev], x[ev], "o", ms=7, color=ORANGE, label="2019 (43,4 °C)")
    ax.plot(years, mu(fit, t), color=BLUE, lw=2, label=r"Position $\mu(t)$ ajustée")
    ax.plot(years, rl100, color=BLUE, lw=1.5, ls="--", label="Niveau de retour 100 ans")
    ax.set_xlabel("Année")
    ax.set_ylabel("TX max juin-août (°C)")
    ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def plot_return_levels(T, curves, bands, obs_shifted, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for curve, (lo, hi), obs, label, c in zip(curves, bands, obs_shifted,
                                              ["Climat de 1950", "Climat de 2025"], [BLUE, ORANGE]):
        ax.fill_between(T, lo, hi, color=c, alpha=0.18, lw=0)
        ax.plot(T, curve, color=c, lw=2, label=label)
        obs = np.sort(obs)
        ax.plot(gringorten(len(obs)), obs, "o", ms=3.5, color=c, alpha=0.8)
    ax.axhline(43.4, color=INK, lw=1, ls=":")
    ax.text(T[0] * 1.1, 43.6, "43,4 °C (28 juin 2019)", color=INK, va="bottom")
    ax.set_xscale("log")
    # ax.set_ylim(30, 52)
    ax.set_xlabel("Période de retour (années)")
    ax.set_ylabel("TX max juin-août (°C)")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def plot_qq(x, fit, t, path):
    # résidus ramenés à une gumbel standard : log(1 + xi*z) / xi
    z = (x - mu(fit, t)) / fit["sigma"]
    xi = fit["xi"]
    resid = np.sort(np.log1p(xi * z) / xi if abs(xi) > 1e-6 else z)
    n = len(resid)
    p = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
    theo = -np.log(-np.log(p))

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    lims = [min(theo.min(), resid.min()) - 0.3, max(theo.max(), resid.max()) + 0.3]
    ax.plot(lims, lims, color=GREY, lw=1)
    ax.plot(theo, resid, "o", ms=4, color=BLUE)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Quantiles théoriques (Gumbel)")
    ax.set_ylabel("Résidus standardisés")
    fig.savefig(path)
    plt.close(fig)
