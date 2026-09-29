import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

STATION = 13103001  # salon de provence (base aérienne)
DEPT = 13
GISTEMP_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"


def download(url, dest):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print("téléchargement de", url)
        urllib.request.urlretrieve(url, dest)
    return dest


def meteofrance_urls(dept=DEPT):
    # les noms de fichiers changent tous les ans (previous-1950-2024 etc.)
    # donc je passe par l'api data.gouv au lieu de les mettre en dur
    api = "https://www.data.gouv.fr/api/1/datasets/donnees-climatologiques-de-base-quotidiennes/"
    with urllib.request.urlopen(api) as r:
        resources = json.load(r)["resources"]
    pat = re.compile(rf"/Q_{dept:02d}_.*_RR-T-Vent\.csv\.gz$")
    return sorted(r["url"] for r in resources if pat.search(r["url"]))


def load_station(station=STATION, dept=DEPT):
    tmp = []
    for url in meteofrance_urls(dept):
        f = download(url, DATA_DIR / "raw" / url.split("/")[-1])
        df = pd.read_csv(f, sep=";", usecols=["NUM_POSTE", "AAAAMMJJ", "TX", "QTX"],
                         dtype={"NUM_POSTE": "int64", "AAAAMMJJ": "str"})
        tmp.append(df[df.NUM_POSTE == station])

    df = pd.concat(tmp).dropna(subset=["TX"])
    df["date"] = pd.to_datetime(df.AAAAMMJJ, format="%Y%m%d")
    df = df.rename(columns={"TX": "tx", "QTX": "qtx"})
    return df[["date", "tx", "qtx"]].drop_duplicates("date").sort_values("date")


def summer_maxima(daily, min_coverage=0.9):
    # tous les max annuels de la station sont en juin-août, donc on travaille
    # sur l'été : ça permet de garder 1939, 1947 et 2026 (hivers incomplets)
    ete = daily[daily.date.dt.month.isin([6, 7, 8])]
    g = ete.groupby(ete.date.dt.year)["tx"]
    maxi = g.max()[g.count() / 92 >= min_coverage]
    maxi.index.name = "year"
    return maxi.rename("tx_max")


def load_gmst(window=4):
    # anomalie globale gistemp, moyenne sur les 4 dernières années comme WWA
    f = download(GISTEMP_URL, DATA_DIR / "raw" / "gistemp_global.csv")
    raw = pd.read_csv(f, skiprows=1, na_values="***", index_col="Year")
    mois = raw.iloc[:, :12]
    # année en cours pas finie : moyenne des mois dispo s'il y en a au moins 6
    annual = mois.mean(axis=1)[mois.count(axis=1) >= 6]
    return annual.rolling(window).mean().dropna().rename("gmst")


def load_dataset():
    df = pd.concat([summer_maxima(load_station()), load_gmst()], axis=1, join="inner")
    return df.astype(float)
