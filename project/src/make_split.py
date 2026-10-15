"""
Frozen train / test split, plus spatial CV folds inside the training set.

Step 1  TEST SET   : whole spatial blocks (and/or the latest seasons) are held out.
                     It depends only on the block layout, the seed and the panel row
                     counts, never on the labels or on the CV settings, so it stays the
                     same when you later change BUFFER_KM, K_FOLDS or CV_BLOCK_KM.
Step 2  BUFFER     : training pixels within BUFFER_KM of a test pixel are set aside
                     (role="buffer"), because neighbouring cells are near-copies.
Step 3  CV FOLDS   : inside the training rows, CV_BLOCK_KM blocks are dealt into K folds
                     with similar row counts. Fold k is the validation set in round k.

Input : data/processed/panel_same_date_model/   (pixel_id, date, fire, season columns)
        data/processed/ca_grid_5km.parquet      (pixel_id, lat, lon)
Output: data/processed/split.parquet            one row per (pixel_id, date):
            role      "train" | "test" | "buffer" | "unused"
            cv_fold   0..K-1 for train rows, -1 otherwise
            block_test, block_cv, x_km, y_km
        data/processed/split_params.json        settings and test blocks, for the report

Usage:
    python make_split.py                         # spatial hold-out (default)
    SPLIT_MODE=both TEST_FROM_SEASON_YEAR=2023 python make_split.py
    OVERWRITE=1 BUFFER_KM=20 python make_split.py   # re-run; reports if the test set changed

Using the folds with scikit-learn:
    from make_split import cv_splits
    frame = model_table.merge(split, on=["pixel_id", "date"])
    train = frame[frame["role"] == "train"].reset_index(drop=True)
    folds = list(cv_splits(train, buffer_km=10))     # (fit_idx, val_idx) pairs
    # e.g. cross_val_score(model, X, y, cv=folds)  or  for fit_idx, val_idx in folds: ...
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ------------------------------- settings --------------------------------------
ROOT = Path(__file__).resolve().parent
PROC = Path(os.environ.get("PROC_DIR", ROOT / "data/processed"))
MODEL_DIR = PROC / "panel_same_date_model"
GRID_PATH = PROC / "ca_grid_5km.parquet"
OUT_PATH = PROC / "split.parquet"
PARAMS_PATH = PROC / "split_params.json"

# "spatial": hold out whole blocks | "temporal": hold out the latest seasons |
# "both": test = unseen blocks AND latest seasons (strictest, discards the most data)
SPLIT_MODE = os.environ.get("SPLIT_MODE", "spatial")
TEST_BLOCK_KM = float(os.environ.get("TEST_BLOCK_KM", "60"))
TEST_FRAC = float(os.environ.get("TEST_FRAC", "0.2"))          # share of ROWS held out
TEST_FROM_SEASON_YEAR = int(os.environ.get("TEST_FROM_SEASON_YEAR", "2023"))
BUFFER_KM = float(os.environ.get("BUFFER_KM", "10"))           # train/test gap
K_FOLDS = int(os.environ.get("K_FOLDS", "5"))
CV_BLOCK_KM = float(os.environ.get("CV_BLOCK_KM", "30"))       # set from the correlogram range
TEST_SEED = int(os.environ.get("TEST_SEED", "42"))
CV_SEED = int(os.environ.get("CV_SEED", "7"))
GRID_CRS = os.environ.get("GRID_CRS") or None                  # e.g. "EPSG:3310" (needs pyproj)
OVERWRITE = os.environ.get("OVERWRITE", "0") == "1"
# --------------------------------------------------------------------------------


def to_km(lat: np.ndarray, lon: np.ndarray):
    """Coordinates in km. Exact in GRID_CRS if given, otherwise a flat approximation."""
    if GRID_CRS:
        from pyproj import Transformer

        x, y = Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True).transform(lon, lat)
        return np.asarray(x) / 1000.0, np.asarray(y) / 1000.0
    lat0, lon0 = lat.mean(), lon.mean()
    return (lon - lon0) * 111.32 * np.cos(np.radians(lat0)), (lat - lat0) * 110.57


def load_panel() -> pd.DataFrame:
    cols = ["pixel_id", "date", "fire", "season", "season_year", "season_complete"]
    df = pd.read_parquet(MODEL_DIR, columns=cols)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    grid = pd.read_parquet(GRID_PATH, columns=["pixel_id", "lat", "lon"]).drop_duplicates("pixel_id")
    df = df.merge(grid, on="pixel_id", how="left", validate="m:1")
    if df["lat"].isna().any():
        raise ValueError(f"{df.loc[df['lat'].isna(), 'pixel_id'].nunique()} pixels have no coordinates")
    return df


def choose_test_blocks(pix: pd.DataFrame) -> list:
    """Random blocks until about TEST_FRAC of the rows are covered (uses row counts only)."""
    rng = np.random.default_rng(TEST_SEED)
    rows_per_block = pix.groupby("block_test")["n_rows"].sum()
    total, chosen, covered = rows_per_block.sum(), [], 0
    for b in rng.permutation(rows_per_block.index.values):
        if covered / total >= TEST_FRAC:
            break
        chosen.append(b)
        covered += rows_per_block[b]
    return sorted(chosen)


def assign_cv_folds(train: pd.DataFrame) -> dict:
    """Deal CV blocks into K_FOLDS folds so each fold has a similar number of rows."""
    rng = np.random.default_rng(CV_SEED)
    size = train.groupby("block_cv").size()
    blocks = list(rng.permutation(size.index.values))
    blocks.sort(key=lambda b: -size[b])            # big blocks first (stable, random ties)
    load = np.zeros(K_FOLDS)
    fold_of = {}
    for b in blocks:
        k = int(load.argmin())
        fold_of[b] = k
        load[k] += size[b]
    return fold_of


def make_split() -> tuple[pd.DataFrame, dict]:
    df = load_panel()

    pix = df.groupby("pixel_id").agg(n_rows=("fire", "size"), lat=("lat", "first"), lon=("lon", "first"))
    pix = pix.reset_index()
    pix["x_km"], pix["y_km"] = to_km(pix["lat"].values, pix["lon"].values)
    x0, y0 = pix["x_km"].min(), pix["y_km"].min()

    def block_id(size_km):
        bx = ((pix["x_km"] - x0) // size_km).astype(int)
        by = ((pix["y_km"] - y0) // size_km).astype(int)
        return bx.astype(str) + "_" + by.astype(str)

    pix["block_test"] = block_id(TEST_BLOCK_KM)
    pix["block_cv"] = block_id(CV_BLOCK_KM)

    # ---- spatial test blocks and buffer ----
    use_space = SPLIT_MODE in ("spatial", "both")
    test_blocks = choose_test_blocks(pix) if use_space else []
    pix["in_test_block"] = pix["block_test"].isin(test_blocks)
    pix["in_buffer"] = False
    if use_space and BUFFER_KM > 0 and pix["in_test_block"].any():
        tree = cKDTree(pix.loc[pix["in_test_block"], ["x_km", "y_km"]].values)
        dist, _ = tree.query(pix[["x_km", "y_km"]].values)
        pix["in_buffer"] = (~pix["in_test_block"]) & (dist < BUFFER_KM)

    # ---- row-level roles ----
    r = df.merge(pix[["pixel_id", "block_test", "block_cv", "x_km", "y_km", "in_test_block", "in_buffer"]],
                 on="pixel_id", how="left", validate="m:1")

    late = r["season_year"] >= TEST_FROM_SEASON_YEAR

    if SPLIT_MODE == "spatial":
        conds = [r["in_test_block"], r["in_buffer"]]
        r["role"] = np.select(conds, ["test", "buffer"], default="train")
    elif SPLIT_MODE == "temporal":
        r["role"] = np.where(late, "test", "train")
    elif SPLIT_MODE == "both":
        conds = [
            r["in_test_block"] & late,                       # test: unseen blocks, latest seasons
            r["in_buffer"] & ~late,                          # buffer around the test blocks
            ~r["in_test_block"] & ~r["in_buffer"] & ~late,   # train: seen blocks, earlier seasons
        ]
        r["role"] = np.select(conds, ["test", "buffer", "train"], default="unused")
    else:
        raise ValueError("SPLIT_MODE must be spatial, temporal or both")

    # ---- spatial CV folds on training rows only ----
    r["cv_fold"] = -1
    is_train = r["role"] == "train"
    fold_of = assign_cv_folds(r.loc[is_train])
    r.loc[is_train, "cv_fold"] = r.loc[is_train, "block_cv"].map(fold_of).astype(int)

    out = r[["pixel_id", "date", "role", "cv_fold", "block_test", "block_cv", "x_km", "y_km"]].copy()
    r_full = r  # keep fire/season for the printed report
    params = {
        "split_mode": SPLIT_MODE, "test_block_km": TEST_BLOCK_KM, "test_frac": TEST_FRAC,
        "test_from_season_year": TEST_FROM_SEASON_YEAR, "buffer_km": BUFFER_KM,
        "k_folds": K_FOLDS, "cv_block_km": CV_BLOCK_KM, "test_seed": TEST_SEED, "cv_seed": CV_SEED,
        "n_test_blocks": len(test_blocks), "test_blocks": test_blocks,
    }
    return out, params, r_full


def report(r: pd.DataFrame):
    print("\n--- roles ---")
    t = r.groupby("role").agg(rows=("fire", "size"), pixels=("pixel_id", "nunique"), fire_rate=("fire", "mean"))
    t["row_share"] = t["rows"] / len(r)
    print(t.round(3).to_string())
    print("\nrows by role and season:")
    print(pd.crosstab(r["role"], r["season"]).to_string())

    tr = r[r["role"] == "train"]
    print("\n--- CV folds (training rows) ---")
    f = tr.groupby("cv_fold").agg(rows=("fire", "size"), pixels=("pixel_id", "nunique"),
                                  blocks=("block_cv", "nunique"), fire_rate=("fire", "mean"))
    print(f.round(3).to_string())
    print("\nrows per fold and season:")
    print(pd.crosstab(tr["cv_fold"], tr["season"]).to_string())
    if f["blocks"].min() < 5:
        print("\nWARNING: some folds have fewer than 5 blocks; consider a smaller CV_BLOCK_KM or fewer folds.")
    te = r[r["role"] == "test"]
    if len(te):
        print(f"\ntest fire rate {te['fire'].mean():.3f} vs train {tr['fire'].mean():.3f}")


def cv_splits(frame: pd.DataFrame, buffer_km: float = 0.0):
    """Yield (fit_idx, val_idx) positional index pairs for spatial CV.

    `frame` must contain role, cv_fold, x_km, y_km (merge the split file onto your
    table first). Fit rows within buffer_km of the validation fold are dropped, so
    pixels right across a fold border do not leak into validation.
    """
    train = (frame["role"] == "train").values
    folds = frame["cv_fold"].values
    xy = frame[["x_km", "y_km"]].values
    for k in sorted(set(folds[train])):
        val = train & (folds == k)
        fit = train & (folds != k)
        fit_idx = np.flatnonzero(fit)
        if buffer_km > 0 and val.any():
            dist, _ = cKDTree(np.unique(xy[val], axis=0)).query(xy[fit_idx])
            fit_idx = fit_idx[dist >= buffer_km]
        yield fit_idx, np.flatnonzero(val)


def main():
    old = None
    if OUT_PATH.exists():
        if not OVERWRITE:
            raise SystemExit(f"{OUT_PATH} already exists. The test split is meant to stay frozen; "
                             f"set OVERWRITE=1 to rebuild it.")
        old = pd.read_parquet(OUT_PATH, columns=["pixel_id", "date", "role"])

    out, params, r = make_split()
    report(r)

    if old is not None:
        a = set(map(tuple, old.loc[old["role"] == "test", ["pixel_id", "date"]].values.tolist()))
        b = set(map(tuple, out.loc[out["role"] == "test", ["pixel_id", "date"]].values.tolist()))
        print(f"\ntest set identical to the previous split: {a == b} ({len(a):,} vs {len(b):,} rows)")

    out.to_parquet(OUT_PATH, index=False)
    PARAMS_PATH.write_text(json.dumps(params, indent=2))
    print(f"\nsaved {OUT_PATH} and {PARAMS_PATH}")


if __name__ == "__main__":
    main()
