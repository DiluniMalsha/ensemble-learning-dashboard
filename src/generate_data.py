"""
Generates the two *synthetic* Sri Lankan weather datasets used in the project.

1. data/sri_lanka_rain.csv      -> tabular classification ("Will it rain tomorrow?")
2. data/colombo_monthly_rainfall.csv -> monthly time series (for the ARIMA + RF hybrid)

IMPORTANT: these are SIMULATED datasets. They follow the real climatic pattern of
Sri Lanka (South-West monsoon May-Sep hits the wet zone, North-East monsoon Dec-Feb
hits the dry zone / east, inter-monsoon thunderstorms in Mar-Apr and Oct-Nov), but the
numbers are generated, not measured. Replace them with real Department of Meteorology
data if you have access - the app works with any CSV.

Run:  python src/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# district -> (climate zone, mean temp °C, elevation-ish factor)
DISTRICTS = {
    "Colombo": ("wet", 28.0), "Gampaha": ("wet", 27.8), "Galle": ("wet", 27.2),
    "Ratnapura": ("wet", 27.0), "Kandy": ("intermediate", 24.5),
    "Nuwara Eliya": ("wet_hill", 16.0), "Badulla": ("intermediate", 22.5),
    "Kurunegala": ("intermediate", 27.8), "Anuradhapura": ("dry", 28.5),
    "Jaffna": ("dry", 28.8), "Trincomalee": ("dry_east", 29.0),
    "Batticaloa": ("dry_east", 28.6), "Hambantota": ("dry", 28.2),
}


def season_of(month: int) -> str:
    if month in (5, 6, 7, 8, 9):
        return "SW Monsoon"
    if month in (12, 1, 2):
        return "NE Monsoon"
    if month in (3, 4):
        return "First Inter-monsoon"
    return "Second Inter-monsoon"  # Oct, Nov


# extra log-odds of rain for each (zone, season)
SEASON_EFFECT = {
    "wet":          {"SW Monsoon": 1.4, "NE Monsoon": -0.6, "First Inter-monsoon": 0.4, "Second Inter-monsoon": 0.9},
    "wet_hill":     {"SW Monsoon": 1.6, "NE Monsoon": 0.0,  "First Inter-monsoon": 0.4, "Second Inter-monsoon": 0.9},
    "intermediate": {"SW Monsoon": 0.2, "NE Monsoon": 0.5,  "First Inter-monsoon": 0.3, "Second Inter-monsoon": 1.0},
    "dry":          {"SW Monsoon": -1.3, "NE Monsoon": 0.9, "First Inter-monsoon": -0.2, "Second Inter-monsoon": 0.8},
    "dry_east":     {"SW Monsoon": -1.5, "NE Monsoon": 1.5, "First Inter-monsoon": -0.4, "Second Inter-monsoon": 1.1},
}


def make_rain_dataset(n: int = 6000) -> pd.DataFrame:
    districts = RNG.choice(list(DISTRICTS), size=n)
    months = RNG.integers(1, 13, size=n)
    rows = []
    for d, m in zip(districts, months):
        zone, base_t = DISTRICTS[d]
        season = season_of(int(m))
        s_eff = SEASON_EFFECT[zone][season]

        humidity = np.clip(RNG.normal(72 + 8 * s_eff, 9), 35, 100)
        cloud = np.clip(RNG.normal(4 + 1.2 * s_eff + (humidity - 75) * 0.06, 1.6), 0, 8)
        pressure = RNG.normal(1009.5 - 1.2 * s_eff, 2.2)
        temp = RNG.normal(base_t - 0.8 * max(s_eff, 0), 1.4)
        wind = np.clip(RNG.gamma(2.2, 4.5) + 4 * max(s_eff, 0), 0, 60)
        if season == "SW Monsoon":
            wind_dir = RNG.choice(["SW", "W", "S", "NW"], p=[0.55, 0.25, 0.12, 0.08])
        elif season == "NE Monsoon":
            wind_dir = RNG.choice(["NE", "N", "E", "SE"], p=[0.55, 0.2, 0.17, 0.08])
        else:
            wind_dir = RNG.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        rain_today = 0.0
        if RNG.random() < 1 / (1 + np.exp(-(s_eff + (humidity - 78) * 0.08))):
            rain_today = float(np.round(RNG.gamma(1.4, 9 + 6 * max(s_eff, 0)), 1))

        # ---- the "true" (hidden) rain-tomorrow mechanism ----
        logit = (
            -1.2 + 1.1 * s_eff
            + 0.07 * (humidity - 75)
            + 0.35 * (cloud - 4)
            - 0.28 * (pressure - 1009)
            + 0.018 * rain_today
            + 0.6 * (rain_today > 0) * (humidity > 85)        # interaction
            + (0.5 if (zone.startswith("wet") and wind_dir in ("SW", "W")) else 0)
            + (0.5 if (zone.startswith("dry") and wind_dir in ("NE", "E")) else 0)
            + RNG.normal(0, 0.9)                               # unexplained weather noise
        )
        rain_tomorrow = "Yes" if RNG.random() < 1 / (1 + np.exp(-logit)) else "No"

        rows.append({
            "district": d, "month": int(m), "season": season,
            "temperature_c": round(float(temp), 1), "humidity_pct": round(float(humidity), 1),
            "pressure_hpa": round(float(pressure), 1), "wind_speed_kmh": round(float(wind), 1),
            "wind_direction": wind_dir, "cloud_cover_okta": round(float(cloud), 1),
            "rain_today_mm": rain_today, "rain_tomorrow": rain_tomorrow,
        })
    df = pd.DataFrame(rows)
    # realistic messiness: ~3% missing values in some sensor columns
    for col in ["humidity_pct", "pressure_hpa", "wind_speed_kmh", "cloud_cover_okta"]:
        mask = RNG.random(n) < 0.03
        df.loc[mask, col] = np.nan
    return df


def make_monthly_rainfall(start="1990-01", end="2025-12") -> pd.DataFrame:
    idx = pd.period_range(start, end, freq="M")
    # approximate Colombo monthly climatology (mm): two peaks (May & Oct)
    clim = {1: 70, 2: 75, 3: 120, 4: 245, 5: 380, 6: 220, 7: 135, 8: 115, 9: 240, 10: 365, 11: 310, 12: 165}
    n = len(idx)
    ar = np.zeros(n)
    eps = RNG.normal(0, 1, n)
    for t in range(1, n):  # persistent anomalies (AR(1)) e.g. ENSO-like wet/dry spells
        ar[t] = 0.7 * ar[t - 1] + eps[t]
    trend = np.linspace(0, 25, n)  # slight upward trend
    vals = []
    for i, p in enumerate(idx):
        base = clim[p.month]
        # non-linear effect: anomalies hit monsoon months harder (what ARIMA struggles with)
        boost = 1.0 + (0.6 if p.month in (5, 10, 11) else 0.2) * np.tanh(ar[i])
        v = base * boost + trend[i] + RNG.normal(0, 0.12 * base)
        vals.append(max(5.0, round(float(v), 1)))
    return pd.DataFrame({"month": idx.to_timestamp().strftime("%Y-%m-%d"), "rainfall_mm": vals})


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    rain = make_rain_dataset()
    rain.to_csv(DATA_DIR / "sri_lanka_rain.csv", index=False)
    print("sri_lanka_rain.csv", rain.shape, rain["rain_tomorrow"].value_counts().to_dict())
    ts = make_monthly_rainfall()
    ts.to_csv(DATA_DIR / "colombo_monthly_rainfall.csv", index=False)
    print("colombo_monthly_rainfall.csv", ts.shape)
