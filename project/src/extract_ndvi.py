"""
Extract pre-event NDVI from Google Earth Engine for a (pixel_id, date) panel.

Input : data/processed/panel_same_date/       (date, pixel_id, fire, year)
        data/processed/ca_grid_5km.parquet    (pixel_id, lat, lon, ...)
Output: data/processed/panel_same_date_ndvi/year=YYYY/part-0.parquet
        columns: pixel_id, date, ndvi, n_images
        (join to the climate panel on pixel_id + date later)

Setup:
    pip install earthengine-api pandas pyarrow tqdm python-dotenv
    earthengine authenticate            # one-time per machine

Usage (settings via environment variables or a .env file):
    GOOGLE_CLOUD_PROJECT=my-project YEARS=2015 python extract_ndvi.py   # one-year test
    GOOGLE_CLOUD_PROJECT=my-project python extract_ndvi.py              # all years
    ... YEARS=2015 TEST_N=50 python extract_ndvi.py                     # tiny smoke test
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import pandas as pd
from tqdm import tqdm

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ------------------- machine-specific settings (environment) -------------------
EE_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]  # required; fails loudly if missing
PANEL_DIR = Path(os.environ.get("PANEL_DIR", "project/data/processed/panel_same_date"))
GRID_PATH = Path(os.environ.get("GRID_PATH", "project/data/processed/ca_grid_5km.parquet"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "project/data/processed/panel_same_date_ndvi"))
YEARS = [int(y) for y in os.environ["YEARS"].split(",")] if os.environ.get("YEARS") else None
TEST_N = int(os.environ["TEST_N"]) if os.environ.get("TEST_N") else None
OVERWRITE = os.environ.get("OVERWRITE", "0") == "1"  # redo years that already have output

# ------------------------- analysis parameters (in code) -----------------------
ID_COL, DATE_COL, YEAR_COL = "pixel_id", "date", "year"
LAT_COL, LON_COL = "lat", "lon"

# "cell_mean": mean NDVI over the 5 km cell footprint (square around the centroid)
# "centroid" : value of the 250 m MODIS pixel at the centroid only
SAMPLING = "cell_mean"
CELL_SIZE_M = 5000

COLLECTION = "MODIS/061/MOD13Q1"  # 250 m, 16-day composites
BAND = "NDVI"
SCALE_FACTOR = 0.0001
SCALE_M = 250

# MODIS 16-day composites are stamped with their START date and cover 16 days.
# To use only data fully BEFORE the event date d, we take composites whose start s
# satisfies s + 16 <= d. N_COMPOSITES=1 -> the most recent complete composite.
COMPOSITE_DAYS = 16
N_COMPOSITES = 1

CHUNK_SIZE = 1000  # points per request (keep <= 5000)
N_WORKERS = 4
MAX_RETRIES = 5
# -------------------------------------------------------------------------------


def make_extractor():
    coll = (
        ee.ImageCollection(COLLECTION)
        .select(BAND)
        .map(lambda im: im.updateMask(im.gte(-2000)))  # MODIS fill value is -3000
    )
    empty = ee.Image.constant(0).rename(BAND).updateMask(0)

    def extract(f):
        d = ee.Date(f.get("date_ms"))
        start = d.advance(-COMPOSITE_DAYS * (N_COMPOSITES + 1) + 1, "day")
        end = d.advance(-COMPOSITE_DAYS + 1, "day")  # filterDate end is exclusive
        win = coll.filterDate(start, end)
        n_img = win.size()
        img = ee.Image(ee.Algorithms.If(n_img.gt(0), win.mean(), empty))

        region = f.geometry()
        if SAMPLING == "cell_mean":
            region = region.buffer(CELL_SIZE_M / 2).bounds()

        val = ee.Dictionary(
            img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=SCALE_M,
                maxPixels=int(1e8),
            )
        ).get(BAND, None)
        return ee.Feature(None, {"rid": f.get("rid"), "ndvi_raw": val, "n_images": n_img})

    return extract


def load_points(year: int, grid: pd.DataFrame) -> pd.DataFrame:
    """Unique (pixel_id, date) pairs for one year, with coordinates."""
    panel = pd.read_parquet(
        PANEL_DIR, filters=[(YEAR_COL, "==", year)], columns=[ID_COL, DATE_COL]
    )
    panel[DATE_COL] = pd.to_datetime(panel[DATE_COL]).dt.normalize()
    pts = panel.drop_duplicates([ID_COL, DATE_COL]).merge(grid, on=ID_COL, how="left")

    n_missing = pts[LAT_COL].isna().sum()
    if n_missing:
        print(f"  WARNING: {n_missing} rows have no coordinates in the grid file; dropping")
        pts = pts.dropna(subset=[LAT_COL, LON_COL])

    if TEST_N:
        pts = pts.head(TEST_N)
    pts = pts.reset_index(drop=True)
    pts["rid"] = pts.index.astype("int64")
    pts["date_ms"] = (
        (pts[DATE_COL] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
    ).astype("int64")
    return pts


def run_chunk(path: Path, chunk: pd.DataFrame, extract) -> pd.DataFrame:
    if path.exists():  # checkpoint from an earlier run
        return pd.read_parquet(path)

    feats = [
        ee.Feature(
            ee.Geometry.Point([float(r.lon), float(r.lat)]),
            {"rid": int(r.rid), "date_ms": int(r.date_ms)},
        )
        for r in chunk.itertuples(index=False)
    ]
    for attempt in range(MAX_RETRIES):
        try:
            info = ee.FeatureCollection(feats).map(extract).getInfo()
            res = pd.DataFrame([ft["properties"] for ft in info["features"]])
            res.to_parquet(path)
            return res
        except Exception as e:  # rate limits, timeouts, ...
            wait = 2**attempt * 5
            print(f"[{path.name}] error: {e} -> retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"{path.name} failed after {MAX_RETRIES} retries")


def process_year(year: int, grid: pd.DataFrame, extract):
    out_root = Path(str(OUT_DIR) + "_test") if TEST_N else OUT_DIR
    out_file = out_root / f"year={year}" / "part-0.parquet"
    if out_file.exists() and not OVERWRITE:
        print(f"{year}: output exists, skipping (set OVERWRITE=1 to redo)")
        return

    pts = load_points(year, grid)
    print(f"{year}: {len(pts):,} unique (pixel, date) pairs, {pts[ID_COL].nunique():,} pixels")
    if pts.empty:
        return

    ckpt_dir = out_root / "_chunks" / f"year={year}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    chunks = [pts.iloc[i : i + CHUNK_SIZE] for i in range(0, len(pts), CHUNK_SIZE)]

    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [
            ex.submit(run_chunk, ckpt_dir / f"chunk_{i:05d}.parquet", c, extract)
            for i, c in enumerate(chunks)
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=str(year)):
            results.append(fut.result())

    res = pd.concat(results, ignore_index=True)
    res["ndvi"] = res["ndvi_raw"].astype("float64") * SCALE_FACTOR
    out = pts[["rid", ID_COL, DATE_COL]].merge(res[["rid", "ndvi", "n_images"]], on="rid")
    out = out.drop(columns="rid")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_file, index=False)

    miss = out["ndvi"].isna().mean()
    print(f"{year}: saved {out_file} | missing NDVI {miss:.1%}")
    print(out["ndvi"].describe().round(3).to_string())


def main():
    ee.Initialize(project=EE_PROJECT)
    grid = pd.read_parquet(GRID_PATH, columns=[ID_COL, LAT_COL, LON_COL])

    years = YEARS
    if years is None:
        years = sorted(pd.read_parquet(PANEL_DIR, columns=[YEAR_COL])[YEAR_COL].unique())
    extract = make_extractor()
    for y in years:
        process_year(int(y), grid, extract)


if __name__ == "__main__":
    main()
