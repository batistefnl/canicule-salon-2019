"""Analyse de la canicule de juin 2019 à Salon-de-Provence.

usage : python analyse.py [n_boot]   (1000 par défaut, 100 pour tester vite)
"""
import json
import sys
from pathlib import Path

import numpy as np

from data import load_dataset
from gev import borne_sup, bootstrap, fit_gev, lrt, return_level, return_period
from plots import plot_qq, plot_return_levels, plot_series

EVENT = 43.4  # TX du 28/06/2019
T = np.logspace(0.05, 3, 60)  # périodes de retour de ~1,1 à 1000 ans


def ci(samples, level=0.95):
    a = (1 - level) / 2
    # method="inverted_cdf" sinon np.quantile renvoie nan quand il y a des inf
    return np.quantile(samples, [a, 1 - a], method="inverted_cdf", axis=0)


def fmt(p):
    if not np.isfinite(p):
        return "infinie"
    if p >= 1000:
        return f"{p:,.0f}".replace(",", " ")
    return f"{p:.0f}"


def main():
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    df = load_dataset()
    # print(df.loc[2015:])
    years = df.index.values
    x, t = df.tx_max.values, df.gmst.values
    g50, g25 = df.loc[1950, "gmst"], df.loc[2025, "gmst"]

    fit0 = fit_gev(x)
    fit1 = fit_gev(x, t)
    D, pval = lrt(fit0, fit1)

    rp50 = float(return_period(fit1, EVENT, g50))
    rp25 = float(return_period(fit1, EVENT, g25))
    dT = g25 - g50

    def stat(f):
        return np.concatenate([
            [f["alpha"], f["xi"], f["alpha"] * dT,
             return_period(f, EVENT, g50), return_period(f, EVENT, g25)],
            return_level(f, T, g50),
            return_level(f, T, g25),
        ])

    # c'est ça qui prend du temps (~2-3 min pour 1000)
    # TODO essayer la vraisemblance profilée pour les IC, ce serait plus rapide
    boot = bootstrap(fit1, t, stat, n=n_boot)
    k = len(T)
    alpha_b, xi_b, dI_b, rp50_b, rp25_b = boot[:, :5].T
    rl50_b, rl25_b = boot[:, 5:5 + k], boot[:, 5 + k:]
    # inf / inf = nan quand 43.4 est hors support dans les deux climats -> on vire
    with np.errstate(divide="ignore", invalid="ignore"):
        pr_b = rp50_b / rp25_b
    pr_b = pr_b[~np.isnan(pr_b)]

    # sensibilité au choix des années (biais de sélection)
    sens = []
    for name, mask in [("Toutes les années", years > 0),
                       ("Sans 2019", years != 2019),
                       ("Jusqu'à 2018", years < 2019)]:
        f = fit_gev(x[mask], t[mask])
        sens.append({
            "cas": name,
            "n": int(mask.sum()),
            "xi": f["xi"],
            "alpha": f["alpha"],
            "periode_retour_1950": float(return_period(f, EVENT, g50)),
            "periode_retour_2025": float(return_period(f, EVENT, g25)),
            "borne_sup_2025": float(borne_sup(f, g25)),
        })
    # ce qu'aurait dit un modèle stationnaire la veille de l'événement
    f = fit_gev(x[years < 2019])
    sens.append({
        "cas": "Stationnaire, jusqu'à 2018",
        "n": int((years < 2019).sum()),
        "xi": f["xi"],
        "alpha": 0.0,
        "periode_retour_1950": float(return_period(f, EVENT)),
        "periode_retour_2025": float(return_period(f, EVENT)),
        "borne_sup_2025": float(borne_sup(f)),
    })

    res = {
        "n_annees": len(df),
        "periode": [int(years.min()), int(years.max())],
        "gmst_reference": {"1950": float(g50), "2025": float(g25)},
        "stationnaire": {"mu": fit0["mu0"], "sigma": fit0["sigma"], "xi": fit0["xi"], "aic": fit0["aic"]},
        "non_stationnaire": {
            "mu0": fit1["mu0"],
            "alpha": fit1["alpha"],
            "sigma": fit1["sigma"],
            "xi": fit1["xi"],
            "aic": fit1["aic"],
            "alpha_ic95": list(ci(alpha_b)),
            "xi_ic95": list(ci(xi_b)),
        },
        "test_rapport_vraisemblance": {"statistique": D, "p_valeur": pval},
        "evenement": {
            "valeur": EVENT,
            "periode_retour_1950": rp50,
            "periode_retour_1950_ic95": list(ci(rp50_b)),
            "periode_retour_2025": rp25,
            "periode_retour_2025_ic95": list(ci(rp25_b)),
            "rapport_probabilites": rp50 / rp25,
            "rapport_probabilites_ic95": list(ci(pr_b)),
            "delta_intensite": fit1["alpha"] * dT,
            "delta_intensite_ic95": list(ci(dI_b)),
            "borne_sup_2025": float(borne_sup(fit1, g25)),
        },
        "sensibilite": sens,
        "n_boot_valides": len(boot),
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/resultats.json", "w") as fp:
        json.dump(res, fp, indent=2, ensure_ascii=False, default=float)

    Path("figures").mkdir(exist_ok=True)
    plot_series(years, x, fit1, t, return_level(fit1, 100, t), "figures/serie_maxima.png")
    # observations ramenées à chaque climat (seul mu bouge donc c'est juste une translation)
    shifted = [x + fit1["alpha"] * (g - t) for g in (g50, g25)]
    plot_return_levels(T, [return_level(fit1, T, g50), return_level(fit1, T, g25)],
                       [ci(rl50_b), ci(rl25_b)], shifted, "figures/niveaux_retour.png")
    plot_qq(x, fit1, t, "figures/qqplot.png")

    print(f"\n{len(df)} étés entre {years.min()} et {years.max()}")
    print(f"Test du rapport de vraisemblance : D = {D:.1f}, p = {pval:.1e}")
    print(f"AIC stationnaire {fit0['aic']:.1f}, non stationnaire {fit1['aic']:.1f}")
    lo, hi = ci(alpha_b)
    print(f"alpha = {fit1['alpha']:.2f} °C/°C  [{lo:.2f}, {hi:.2f}]")
    lo, hi = ci(xi_b)
    print(f"xi    = {fit1['xi']:.3f}  [{lo:.3f}, {hi:.3f}]")
    print(f"sigma = {fit1['sigma']:.2f} °C")
    for c, rp, b in [("1950", rp50, rp50_b), ("2025", rp25, rp25_b)]:
        lo, hi = ci(b)
        print(f"Période de retour de {EVENT} °C, climat {c} : {fmt(rp)} ans  [{fmt(lo)}, {fmt(hi)}]")
    lo, hi = ci(pr_b)
    print(f"Rapport de probabilités : {fmt(rp50 / rp25)}  [{fmt(lo)}, {fmt(hi)}]")
    lo, hi = ci(dI_b)
    print(f"Écart d'intensité 1950 -> 2025 : {fit1['alpha'] * dT:.1f} °C  [{lo:.1f}, {hi:.1f}]")
    print(f"Borne supérieure (climat 2025) : {borne_sup(fit1, g25):.1f} °C")
    print("\nSensibilité :")
    for r in sens:
        print(f"  {r['cas']:<28} xi = {r['xi']:+.3f}  "
              f"T(1950) = {fmt(r['periode_retour_1950']):>10}  "
              f"T(2025) = {fmt(r['periode_retour_2025']):>8}  "
              f"borne 2025 = {r['borne_sup_2025']:.1f} °C")
    print(f"\n({len(boot)} réplications bootstrap valides)")


if __name__ == "__main__":
    main()
