from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from pyproj import CRS

# california wildfire dataset

cal_fire_dir = "./data/California_Historic_Fire_Perimeters_-4891938132824355098.geojson"
cal_fire_gdf = gpd.read_file(cal_fire_dir)
cal_fire_gdf["ALARM_DATE"] = pd.to_datetime(cal_fire_gdf["ALARM_DATE"], utc=True, errors="coerce")
cal_fire_gdf["CONT_DATE"] = pd.to_datetime(cal_fire_gdf["CONT_DATE"], utc=True, errors="coerce")

# california wildfire data (2015-2024)
cal_fire_10ys_gdf = cal_fire_gdf[(cal_fire_gdf["ALARM_DATE"].dt.year >= 2015) & (cal_fire_gdf["ALARM_DATE"].dt.year < 2025)]



"""
California 5 km pixel-day panel
  fire = 1 -> IGNITION   (pixel intersects a fire footprint on that fire's ALARM_DATE)
  fire = 0 -> NO_FIRE    (pixel not burning and not ignited that day)
  ongoing  -> EXCLUDED   (pixel inside a fire footprint after ALARM_DATE up to CONT_DATE)

Expects `cal_fire_10ys_gdf` (GeoDataFrame with ALARM_DATE, CONT_DATE, geometry) in memory.
Output (Parquet, zstd):
  data/processed/ca_grid_5km.parquet      grid lookup (geometry, centroids, lon/lat)
  data/processed/panel/year=YYYY/...      panel: date, pixel_id (int32), fire (int8)
"""

CRS_CA = "EPSG:3310"
GRID_RES = 5000                  # metres
OUT_DIR = Path("data/processed")
NEG_SAMPLE_FRAC = 1.0            # keep this fraction of NO_FIRE rows (e.g. 0.05 to shrink the panel)
SEED = 42
STATE_URL = "https://www2.census.gov/geo/tiger/TIGER2021/STATE/tl_2021_us_state.zip"

# ----------------------------------------------------------------------------
# 1. Grid
# ----------------------------------------------------------------------------
def make_grid():
    ca = gpd.read_file(STATE_URL)
    ca = ca[ca["STUSPS"] == "CA"]

    # sanity check: does the chosen CRS actually cover California?
    aou = CRS.from_user_input(CRS_CA).area_of_use
    w, s, e, n = ca.to_crs("EPSG:4326").total_bounds
    print(f"CRS {CRS_CA}: {CRS.from_user_input(CRS_CA).name} | area of use: {aou.name if aou else 'unknown'}")
    tol = 0.5  # degrees; TIGER state polygons include offshore water/islands slightly outside the CRS bounds
    if aou is None or not (aou.west - tol <= w and aou.south - tol <= s and aou.east + tol >= e and aou.north + tol >= n):
        print("WARNING: California is not inside this CRS's area of use; grid may be distorted.")

    ca = ca.to_crs(CRS_CA)
    xmin, ymin, xmax, ymax = ca.total_bounds
    xs = xmin + GRID_RES * np.arange(int(np.ceil((xmax - xmin) / GRID_RES)))
    ys = ymin + GRID_RES * np.arange(int(np.ceil((ymax - ymin) / GRID_RES)))
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()

    grid = gpd.GeoDataFrame(geometry=shapely.box(gx, gy, gx + GRID_RES, gy + GRID_RES), crs=CRS_CA)
    grid = grid[grid.intersects(ca.union_all())].reset_index(drop=True)

    grid["pixel_id"] = np.arange(len(grid), dtype="int32")
    grid["pixel_label"] = [f"CA_{i + 1:05d}" for i in range(len(grid))]
    cen = grid.geometry.centroid
    grid["x"], grid["y"] = cen.x, cen.y
    cen_ll = cen.to_crs("EPSG:4326")
    grid["lon"], grid["lat"] = cen_ll.x, cen_ll.y
    print(f"Grid cells: {len(grid)}")
    return grid


# ----------------------------------------------------------------------------
# 2. Ignition / ongoing matrices  (days x pixels, boolean)
# ----------------------------------------------------------------------------
def build_matrices(fires, grid, start=None, end=None):
    fires = fires.to_crs(grid.crs)
    alarm = pd.to_datetime(fires["ALARM_DATE"]).dt.tz_localize(None).dt.normalize()
    cont = pd.to_datetime(fires["CONT_DATE"]).dt.tz_localize(None).dt.normalize()
    cont = cont.fillna(alarm)               # missing containment -> 1-day fire
    cont = cont.where(cont >= alarm, alarm)  # bad order -> 1-day fire

    start = pd.Timestamp(start) if start is not None else alarm.min()
    end = pd.Timestamp(end) if end is not None else alarm.max()
    dates = pd.date_range(start, end, freq="D")
    n_days, n_pix = len(dates), len(grid)

    keep = alarm.notna() & (alarm <= end)
    f = fires.loc[keep, ["geometry"]].assign(alarm=alarm[keep], cont=cont[keep]).reset_index(drop=True)

    # one spatial join for all fires: fire index -> grid index
    pairs = gpd.sjoin(f[["geometry"]], grid[["geometry"]], how="inner", predicate="intersects")

    ign = np.zeros((n_days, n_pix), dtype=bool)
    ong = np.zeros((n_days, n_pix), dtype=bool)

    for fire_idx, pix in pairs.groupby(level=0)["index_right"]:
        pix = pix.to_numpy()
        a = (f.at[fire_idx, "alarm"] - start).days
        c = min((f.at[fire_idx, "cont"] - start).days, n_days - 1)
        if 0 <= a < n_days:
            ign[a, pix] = True
        lo = max(a + 1, 0)
        if c >= lo:
            ong[lo:c + 1, pix] = True

    print(f"Days: {n_days} | ignition pixel-days: {int(ign.sum())} | ongoing pixel-days: {int(ong.sum())}")
    return ign, ong, dates


# ----------------------------------------------------------------------------
# 3. controls only on ignition dates, randomly sampled per date
# ----------------------------------------------------------------------------
def write_panel_same_date_controls(ign, ong, dates, out_dir, ratio=5, n_per_date=None, seed=42):
    """
    For every date with >=1 ignition, keep all ignition pixels (fire=1) and randomly draw
    NO_FIRE pixels (fire=0) from that same date.

    ratio      : controls per ignition pixel on that date (default 5:1)
    n_per_date : if set, draw this fixed number of controls per date instead of using `ratio`
    Controls never include ignition pixels or ongoing-fire pixels.
    """
    rng = np.random.default_rng(seed)
    pos = ign & ~ong
    neg_ok = ~ign & ~ong

    d_all, p_all, y_all = [], [], []
    for d in np.flatnonzero(pos.any(axis=1)):
        pos_pix = np.flatnonzero(pos[d])
        pool = np.flatnonzero(neg_ok[d])
        k = n_per_date if n_per_date is not None else ratio * len(pos_pix)
        ctrl_pix = rng.choice(pool, size=min(k, len(pool)), replace=False)

        pix = np.concatenate([pos_pix, ctrl_pix])
        d_all.append(np.full(len(pix), d, dtype="int32"))
        p_all.append(pix.astype("int32"))
        y_all.append(np.concatenate([np.ones(len(pos_pix), "int8"), np.zeros(len(ctrl_pix), "int8")]))

    df = pd.DataFrame({
        "date": dates.values[np.concatenate(d_all)],
        "pixel_id": np.concatenate(p_all),
        "fire": np.concatenate(y_all),
    })
    df["year"] = pd.DatetimeIndex(df["date"]).year.astype("int16")

    df.to_parquet(out_dir / "panel_same_date", partition_cols=["year"],
                  engine="pyarrow", compression="zstd", index=False)
    print(f"Dates: {df['date'].nunique()} | positives: {int(df['fire'].sum()):,} | "
          f"controls: {int((df['fire'] == 0).sum()):,}")
    return df


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

ca_grid_gdf = make_grid()
ca_grid_gdf.to_parquet(OUT_DIR / "ca_grid_5km.parquet")

ign, ong, dates = build_matrices(cal_fire_10ys_gdf, ca_grid_gdf)   # optionally start="2015-01-01", end="2024-12-31"
write_panel(ign, ong, dates, OUT_DIR, neg_frac=NEG_SAMPLE_FRAC, seed=SEED)          # all days
panel = write_panel_same_date_controls(ign, ong, dates, OUT_DIR, ratio=2, seed=SEED)  # ignition dates only

# Reading the panel:
#   panel = pd.read_parquet("data/processed/panel")                              # 'year' comes back as a column
#   panel = pd.read_parquet("data/processed/panel", filters=[("year", "==", 2020)])
#   panel = panel.merge(climate_df, on=["pixel_id", "date"], how="left")