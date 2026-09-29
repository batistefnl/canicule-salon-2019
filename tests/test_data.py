import numpy as np
import pandas as pd

from data import summer_maxima


def test_summer_maxima_drops_incomplete_summers():
    dates = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    daily = pd.DataFrame({"date": dates, "tx": 20.0})
    daily.loc[daily["date"] == "2000-07-14", "tx"] = 38.0
    daily.loc[daily["date"] == "2000-12-01", "tx"] = 50.0  # hors été, ignoré
    # 2001 : il manque tout le mois de juillet
    daily = daily[~((daily["date"].dt.year == 2001) & (daily["date"].dt.month == 7))]

    maxima = summer_maxima(daily)
    assert list(maxima.index) == [2000]
    assert maxima.loc[2000] == 38.0
