"""Demo data generator and CSV preparation."""
import numpy as np
import pandas as pd

CATEGORIES = {"Technology": 1.4, "Furniture": 1.0, "Office Supplies": 0.8}
REGIONS = {"North": 1.1, "South": 0.9, "East": 1.0, "West": 1.2}
MONTH_SEASONALITY = [0.85, 0.8, 0.95, 1.0, 1.0, 1.05, 0.95, 1.0, 1.1, 1.15, 1.3, 1.45]


def generate_sales(seed: int = 42) -> pd.DataFrame:
    """Daily sales by category/region with trend, yearly seasonality and noise."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2022-01-01", "2025-12-31", freq="D")
    rows = []
    for i, d in enumerate(days):
        trend = 1 + 0.0004 * i
        season = MONTH_SEASONALITY[d.month - 1]
        weekday = 1.15 if d.weekday() >= 5 else 1.0
        for cat, cw in CATEGORIES.items():
            for reg, rw in REGIONS.items():
                base = 1000 * cw * rw * trend * season * weekday
                rows.append((d, cat, reg, max(base * rng.normal(1, 0.12), 0)))
    return pd.DataFrame(rows, columns=["order_date", "category", "region", "sales"])


def prepare(raw, date_col, sales_col, cat_col=None, reg_col=None) -> pd.DataFrame:
    """Standardise an uploaded CSV into order_date / sales / category / region."""
    df = pd.DataFrame({
        "order_date": pd.to_datetime(raw[date_col], errors="coerce"),
        "sales": pd.to_numeric(raw[sales_col], errors="coerce"),
        "category": raw[cat_col].astype(str) if cat_col else "All",
        "region": raw[reg_col].astype(str) if reg_col else "All",
    })
    return df.dropna(subset=["order_date", "sales"])
