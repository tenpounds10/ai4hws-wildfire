"""
Year-matched land cover fractions per 5 km cell, via Google Earth Engine.

For an event in year Y, the land cover map from year Y - LC_LAG is used
(default LAG=1: a 2021 fire uses 2020 land cover). If the product has no map for
that year (e.g. NLCD), the latest available year <= Y - LAG is used. If the target
is earlier than the first map, the first map is used; compare `lc_year` with
`year - LAG` to find and drop those rows.

Input : data/processed/panel_same_date/       (pixel_id, year, ...)
        data/processed/ca_grid_5km.parquet    (pixel_id, lat, lon, ...)
Output: data/processed/panel_same_date_landcover_<product>/year=YYYY/part-0.parquet
        columns: pixel_id, lc_year, lc_<class>..., lc_sum, valid_frac
        Only the pixels present in the panel for that year are extracted.
        Join to the panel on (pixel_id, year).

Usage:
    GOOGLE_CLOUD_PROJECT=my-project YEARS=2015 python extract_landcover_by_year.py   # one-year test
    GOOGLE_CLOUD_PROJECT=my-project python extract_landcover_by_year.py              # all years
    LC_PRODUCT=nlcd LC_LAG=0 ... python extract_landcover_by_year.py
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
EE_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]  # required
# ROOT = Path(__file__).resolve().parent
PRODUCT = os.environ.get("LC_PRODUCT", "modis")  # "modis" or "nlcd"
LC_LAG = int(os.environ.get("LC_LAG", "1"))  # land cover year = event year - LAG
PANEL_DIR = Path(os.environ.get("PANEL_DIR", "project/data/processed/panel_same_date"))
GRID_PATH = Path(os.environ.get("GRID_PATH", "project/data/processed/ca_grid_5km.parquet"))
OUT_DIR = Path(
    os.environ.get("OUT_DIR", f"project/data/processed/panel_same_date_landcover_{PRODUCT}")
)
YEARS = [int(y) for y in os.environ["YEARS"].split(",")] if os.environ.get("YEARS") else None
TEST_N = int(os.environ["TEST_N"]) if os.environ.get("TEST_N") else None
OVERWRITE = os.environ.get("OVERWRITE", "0") == "1"
# CRS of your 5 km grid (e.g. "EPSG:3310"). If set, cells are exact 5 km squares in
# that CRS (needs pyproj); otherwise ~5 km lat/lon-aligned rectangles around centroids.
GRID_CRS = os.environ.get("GRID_CRS") or None

# ------------------------- analysis parameters (in code) -----------------------
ID_COL, YEAR_COL, LAT_COL, LON_COL = "pixel_id", "year", "lat", "lon"
CELL_SIZE_M = 5000

PRODUCTS = {
    # MODIS MCD12Q1, IGBP classification, 500 m, annual. Check ID/band in the Data Catalog.
    "modis": dict(
        collection="MODIS/061/MCD12Q1",
        band="LC_Type1",
        scale=500,
        chunk=1000,
        classes={
            1: "evergreen_needle", 2: "evergreen_broad", 3: "decid_needle",
            4: "decid_broad", 5: "mixed_forest", 6: "closed_shrub", 7: "open_shrub",
            8: "woody_savanna", 9: "savanna", 10: "grassland", 11: "wetland",
            12: "cropland", 13: "urban", 14: "crop_mosaic", 15: "snow_ice",
            16: "barren", 17: "water",
        },
    ),
    # USGS NLCD, 30 m, US only, a map every 2-3 years. Check ID/band in the Data Catalog.
    "nlcd": dict(
        collection="USGS/NLCD_RELEASES/2021_REL/NLCD",
        band="landcover",
        scale=30,
        chunk=1000,
        classes={
            11: "water", 12: "snow_ice", 21: "dev_open", 22: "dev_low",
            23: "dev_med", 24: "dev_high", 31: "barren", 41: "decid_forest",
            42: "evergreen_forest", 43: "mixed_forest", 52: "shrub", 71: "grass",
            81: "pasture", 82: "crop", 90: "woody_wetland", 95: "herb_wetland",
        },
    ),
}
CFG = PRODUCTS[PRODUCT]
N_WORKERS = 4
MAX_RETRIES = 5
# -------------------------------------------------------------------------------


def available_years() -> list[int]:
    ms = ee.ImageCollection(CFG["collection"]).aggregate_array("system:time_start").getInfo()
    return sorted({int(y) for y in pd.to_datetime(ms, unit="ms").year})


def pick_lc_year(target: int, avail: list[int]) -> int:
    eligible = [y for y in avail if y <= target]
    return max(eligible) if eligible else min(avail)


def indicator_image(lc_year: int) -> ee.Image:
    lc = (
        ee.ImageCollection(CFG["collection"])
        .filterDate(f"{lc_year}-01-01", f"{lc_year + 1}-01-01")
        .first()
        .select(CFG["band"])
    )
    bands = [lc.eq(code).rename(f"c{code}") for code in CFG["classes"]]
    bands.append(lc.mask().rename("valid"))  # share of the cell that has data
    return ee.Image.cat(bands)


def make_cell_geometry():
    half = CELL_SIZE_M / 2
    if GRID_CRS:
        from pyproj import Transformer

        tf = Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True)

        def geom(lon, lat):
            x, y = tf.transform(lon, lat)
            return ee.Geometry.Rectangle(
                [x - half, y - half, x + half, y + half], proj=GRID_CRS, geodesic=False
            )
    else:

        def geom(lon, lat):
            return ee.Geometry.Point([lon, lat]).buffer(half).bounds()

    return geom


def run_chunk(path: Path, chunk: pd.DataFrame, img: ee.Image, cell_geom) -> pd.DataFrame:
    if path.exists():  # checkpoint from an earlier run
        return pd.read_parquet(path)

    feats = [
        ee.Feature(cell_geom(float(r.lon), float(r.lat)), {"rid": int(r.rid)})
        for r in chunk.itertuples(index=False)
    ]
    for attempt in range(MAX_RETRIES):
        try:
            res = img.reduceRegions(
                collection=ee.FeatureCollection(feats),
                reducer=ee.Reducer.mean(),
                scale=CFG["scale"],
                tileScale=4,
            ).map(lambda f: ee.Feature(None, f.toDictionary()))  # drop geometry
            info = res.getInfo()
            df = pd.DataFrame([ft["properties"] for ft in info["features"]])
            df.to_parquet(path)
            return df
        except Exception as e:  # rate limits, timeouts, memory, ...
            wait = 2**attempt * 5
            print(f"[{path.name}] error: {e} -> retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"{path.name} failed after {MAX_RETRIES} retries")


def process_year(year: int, grid: pd.DataFrame, avail: list[int], cell_geom):
    out_root = Path(str(OUT_DIR) + "_test") if TEST_N else OUT_DIR
    out_file = out_root / f"year={year}" / "part-0.parquet"
    if out_file.exists() and not OVERWRITE:
        print(f"{year}: output exists, skipping (set OVERWRITE=1 to redo)")
        return

    lc_year = pick_lc_year(year - LC_LAG, avail)

    ids = pd.read_parquet(PANEL_DIR, filters=[(YEAR_COL, "==", year)], columns=[ID_COL])
    pts = ids.drop_duplicates(ID_COL).merge(grid, on=ID_COL, how="left")
    n_missing = pts[LAT_COL].isna().sum()
    if n_missing:
        print(f"  WARNING: {n_missing} pixels have no coordinates; dropping")
        pts = pts.dropna(subset=[LAT_COL, LON_COL])
    if TEST_N:
        pts = pts.head(TEST_N)
    pts = pts.reset_index(drop=True)
    pts["rid"] = pts.index.astype("int64")
    print(f"{year}: {len(pts):,} pixels | land cover year used: {lc_year} (target {year - LC_LAG})")
    if pts.empty:
        return

    img = indicator_image(lc_year)
    ckpt_dir = out_root / "_chunks" / f"year={year}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    size = CFG["chunk"]
    chunks = [pts.iloc[i : i + size] for i in range(0, len(pts), size)]

    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [
            ex.submit(run_chunk, ckpt_dir / f"chunk_{i:05d}.parquet", c, img, cell_geom)
            for i, c in enumerate(chunks)
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=str(year)):
            results.append(fut.result())

    res = pd.concat(results, ignore_index=True)
    out = pts[["rid", ID_COL]].merge(res, on="rid", how="left").drop(columns="rid")
    lc_cols = []
    for code, name in CFG["classes"].items():
        col = f"lc_{name}"
        out[col] = out.pop(f"c{code}") if f"c{code}" in out else float("nan")
        lc_cols.append(col)
    out = out.rename(columns={"valid": "valid_frac"})
    out["lc_sum"] = out[lc_cols].sum(axis=1, min_count=1)  # ~1 where data exists
    out.insert(1, "lc_year", lc_year)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_file, index=False)
    print(
        f"{year}: saved {out_file} | no data: {out['lc_sum'].isna().mean():.2%} "
        f"| valid_frac<0.5: {(out['valid_frac'] < 0.5).mean():.2%}"
    )


def main():
    ee.Initialize(project=EE_PROJECT)

    avail = available_years()
    print(f"{PRODUCT}: maps available for {avail[0]}-{avail[-1]} ({len(avail)} years): {avail}")

    grid = pd.read_parquet(GRID_PATH, columns=[ID_COL, LAT_COL, LON_COL])
    grid = grid.dropna(subset=[LAT_COL, LON_COL]).drop_duplicates(ID_COL)

    years = YEARS
    if years is None:
        years = sorted(pd.read_parquet(PANEL_DIR, columns=[YEAR_COL])[YEAR_COL].unique())

    cell_geom = make_cell_geometry()
    for y in years:
        process_year(int(y), grid, avail, cell_geom)


if __name__ == "__main__":
    main()
