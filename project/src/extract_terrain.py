"""
Cell-mean terrain per 5 km grid cell via Google Earth Engine (SRTM, 30 m).

Why: elevation read at the cell centroid is 0 for coastal cells whose centroid falls
in the water. This averages over the whole cell footprint instead, using only land
pixels (exact-zero DEM pixels are treated as sea and excluded).

Input : data/processed/ca_grid_5km.parquet   (pixel_id, lat, lon, ...)
Output: data/processed/ca_grid_5km_terrain.parquet
        columns: pixel_id, elev_mean, elev_std, slope_mean, dem_land_frac
        Static (one row per pixel); join to the panel on pixel_id.

Usage:
    GOOGLE_CLOUD_PROJECT=my-project TEST_N=50 python extract_terrain.py   # smoke test
    GOOGLE_CLOUD_PROJECT=my-project python extract_terrain.py             # full run
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

EE_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]  # required
ROOT = Path(__file__).resolve().parent
GRID_PATH = Path(os.environ.get("GRID_PATH", ROOT / "data/processed/ca_grid_5km.parquet"))
OUT_PATH = Path(os.environ.get("OUT_PATH", ROOT / "data/processed/ca_grid_5km_terrain.parquet"))
TEST_N = int(os.environ["TEST_N"]) if os.environ.get("TEST_N") else None
GRID_CRS = os.environ.get("GRID_CRS") or None  # e.g. "EPSG:3310"; see extract_landcover.py

ID_COL, LAT_COL, LON_COL = "pixel_id", "lat", "lon"
CELL_SIZE_M = 5000
SCALE_M = 30
CHUNK_SIZE = 1000
N_WORKERS = 4
MAX_RETRIES = 5


def build_image() -> ee.Image:
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    slope = ee.Terrain.slope(dem)  # computed on the full DEM before masking
    land = dem.neq(0)  # exact zero = sea (negative values, e.g. Salton Sea, are kept)
    return ee.Image.cat(
        [
            dem.updateMask(land).rename("elev"),
            slope.updateMask(land).rename("slope"),
            land.unmask(0).rename("dem_land"),  # share of the cell that is land
        ]
    )


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
    reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
    for attempt in range(MAX_RETRIES):
        try:
            res = img.reduceRegions(
                collection=ee.FeatureCollection(feats),
                reducer=reducer,
                scale=SCALE_M,
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


def main():
    ee.Initialize(project=EE_PROJECT)

    grid = pd.read_parquet(GRID_PATH, columns=[ID_COL, LAT_COL, LON_COL])
    grid = grid.dropna(subset=[LAT_COL, LON_COL]).drop_duplicates(ID_COL)
    if TEST_N:
        grid = grid.head(TEST_N)
    grid = grid.reset_index(drop=True)
    grid["rid"] = grid.index.astype("int64")
    print(f"{len(grid):,} cells")

    out_path = OUT_PATH.with_name(OUT_PATH.stem + "_test.parquet") if TEST_N else OUT_PATH
    ckpt_dir = out_path.parent / f"_chunks_{out_path.stem}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    img = build_image()
    cell_geom = make_cell_geometry()
    chunks = [grid.iloc[i : i + CHUNK_SIZE] for i in range(0, len(grid), CHUNK_SIZE)]

    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [
            ex.submit(run_chunk, ckpt_dir / f"chunk_{i:05d}.parquet", c, img, cell_geom)
            for i, c in enumerate(chunks)
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="chunks"):
            results.append(fut.result())

    res = pd.concat(results, ignore_index=True)
    out = grid[["rid", ID_COL]].merge(res, on="rid", how="left").drop(columns="rid")
    out = out.rename(
        columns={
            "elev_mean": "elev_mean",
            "elev_stdDev": "elev_std",
            "slope_mean": "slope_mean",
            "dem_land_mean": "dem_land_frac",
        }
    )[[ID_COL, "elev_mean", "elev_std", "slope_mean", "dem_land_frac"]]
    out.to_parquet(out_path, index=False)

    print(f"Saved {out_path}")
    print(f"cells with no land (elev_mean NaN): {out['elev_mean'].isna().sum():,}")
    print(out.describe().round(2).to_string())


if __name__ == "__main__":
    main()
