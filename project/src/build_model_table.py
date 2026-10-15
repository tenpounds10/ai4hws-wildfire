"""
Join climate + NDVI + land cover + terrain into one model-ready table, one year at a time.

Inputs  (partitioned as <folder>/year=YYYY/):
    panel_same_date_climate/        base table: pixel_id, date, fire, climate features
    panel_same_date_ndvi/           pixel_id, date, ndvi, n_images       -> join on (pixel_id, date)
    panel_same_date_landcover_modis/ pixel_id, lc_year, lc_*, ...        -> join on pixel_id (per year)
    ca_grid_5km_terrain.parquet          pixel_id, terrain_features          -> join on pixel_id (per year)
Output:
    panel_same_date_model/year=YYYY/part-0.parquet

The climate table is the base (left join), so the row count never changes; this is
asserted. Source folders are untouched, so the model table can be rebuilt any time.

Usage:
    python build_model_table.py                 # all years found in the climate folder
    YEARS=2015 python project/build_model_table.py     # test on one year
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


# PROC = Path(os.environ.get("PROC_DIR", "data/processed"))
# CLIMATE_DIR = PROC / "panel_same_date_climate"
# NDVI_DIR = PROC / "panel_same_date_ndvi"
# LC_DIR = PROC / "panel_same_date_landcover_modis"
# OUT_DIR = PROC / "panel_same_date_model"
# YEARS = [int(y) for y in os.environ["YEARS"].split(",")] if os.environ.get("YEARS") else None
# LC_LAG = int(os.environ.get("LC_LAG", "1"))  # must match the land cover run

# ID, DATE = "pixel_id", "date"


ROOT = Path(__file__).resolve().parent
PROC = Path(os.environ.get("PROC_DIR", ROOT / "data/processed"))
CLIMATE_DIR = PROC / "panel_same_date_climate"
NDVI_DIR = PROC / "panel_same_date_ndvi"
LC_DIR = PROC / "panel_same_date_landcover_modis"
TERRAIN_PATH = PROC / "ca_grid_5km_terrain.parquet"
OUT_DIR = PROC / "panel_same_date_model"
YEARS = [int(y) for y in os.environ["YEARS"].split(",")] if os.environ.get("YEARS") else None
LC_LAG = int(os.environ.get("LC_LAG", "1"))  # must match the land cover run
 
ID, DATE = "pixel_id", "date"

WATER_LC_MIN = 0.99     # drop cells that are >= 99% water by land cover
MIN_LAND_FRAC = 0.05    # drop cells with < 5% land according to the DEM


# Seasons: summer = May-Oct, winter = Nov-Apr (winter spans two calendar years).
# DATA_START/END are used to flag seasons that are only partly covered by your data.
DATA_START = pd.Timestamp(os.environ.get("DATA_START", "2015-01-01"))
DATA_END = pd.Timestamp(os.environ.get("DATA_END", "2024-12-31"))


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    month = df[DATE].dt.month
    summer = month.between(5, 10)
    # Jan-Apr belong to the winter that started the previous November
    season_year = df[DATE].dt.year - (month <= 4).astype(int)

    df["season"] = np.where(summer, "summer", "winter")
    df["season_year"] = season_year.astype("int16")
    df["season_id"] = df["season_year"].astype(str) + "_" + df["season"]

    # full date span of each row's season, to flag partial seasons at the data edges
    sy = season_year.astype(str)
    start = pd.to_datetime(sy + np.where(summer, "-05-01", "-11-01"))
    end = pd.to_datetime(np.where(summer, sy + "-10-31", (season_year + 1).astype(str) + "-04-30"))
    df["season_complete"] = (start >= DATA_START) & (end <= DATA_END)
    return df


def read_year(root: Path, year: int) -> pd.DataFrame | None:
    path = root / f"year={year}"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df.drop(columns="year", errors="ignore")  # year is implied by the folder


def norm_date(df: pd.DataFrame) -> pd.DataFrame:
    df[DATE] = pd.to_datetime(df[DATE]).dt.normalize()
    return df


def build_year(year: int, terrain: pd.DataFrame):
    out_file = OUT_DIR / f"year={year}" / "part-0.parquet"

    base = read_year(CLIMATE_DIR, year)
    ndvi = read_year(NDVI_DIR, year)
    lc = read_year(LC_DIR, year)
    for name, df in [("climate", base), ("ndvi", ndvi), ("landcover", lc)]:
        if df is None:
            print(f"{year}: missing {name} partition, skipping year")
            return

    base, ndvi = norm_date(base), norm_date(ndvi)
    ndvi = ndvi.rename(columns={"n_images": "ndvi_n_images"})
    n0 = len(base)

    # guard against silent column collisions (pandas would add _x/_y suffixes)
    for name, df, keys in [("ndvi", ndvi, {ID, DATE}), ("landcover", lc, {ID})]:
        clash = (set(df.columns) - keys) & set(base.columns)
        if clash:
            raise ValueError(f"{year}: {name} columns already in climate table: {clash}")

    m = base.merge(ndvi, on=[ID, DATE], how="left", validate="m:1")
    m = m.merge(lc, on=ID, how="left", validate="m:1")
    assert len(m) == n0, f"{year}: row count changed {n0} -> {len(m)}"
    
    m = m.merge(terrain, on=ID, how="left", validate="m:1")
    assert m["dem_land_frac"].notna().all(), f"{year}: pixels missing from terrain file"

    water = (m["lc_water"] >= WATER_LC_MIN) | (m["dem_land_frac"] < MIN_LAND_FRAC)
    n_drop, n_fire = int(water.sum()), int(m.loc[water, "fire"].sum())
    m = m.loc[~water].reset_index(drop=True)
    print(f"{year}: dropped {n_drop:,} water rows ({n_fire} fires)")
    
    m = add_season(m)

    # smaller files and less RAM when training
    for c in m.select_dtypes("float64").columns:
        m[c] = m[c].astype("float32")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out_file, index=False, compression="zstd")

    leak = (m["lc_year"] > year - LC_LAG).sum()  # land cover newer than intended
    print(
        f"{year}: {n0:,} rows | ndvi missing {m['ndvi'].isna().mean():.1%} "
        f"| landcover missing {m['lc_sum'].isna().mean():.1%} "
        f"| lc_year later than {year - LC_LAG}: {leak:,}"
    )


def main():
    years = YEARS or sorted(
        int(p.name.split("=")[1]) for p in CLIMATE_DIR.glob("year=*")
    )
    # for y in years:
    #     build_year(y)

    terrain = pd.read_parquet(TERRAIN_PATH)
    for y in years:
        build_year(y, terrain)


if __name__ == "__main__":
    main()