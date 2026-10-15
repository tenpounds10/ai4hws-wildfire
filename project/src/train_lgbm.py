"""
Train a LightGBM fire model with spatial cross-validation.

One run = one season model:   SEASON=summer | winter | pooled
  summer / winter : trained and scored on that season's rows only
  pooled          : all rows with an `is_summer` feature; metrics are reported per season,
                    so you can compare it with the stand-alone seasonal models

What it does
  1. Loads the model table + split.parquet. Only role == "train" rows are used for fitting
     and tuning. The test rows are not touched unless EVAL_TEST=1.
  2. Spatial CV: the folds from make_split.py (cv_splits, with a buffer around the
     validation fold).
  3. Random search over LightGBM parameters. Each trial is scored by the mean per-fold ROC-AUC.
     Trial 0 is a sensible default; seasons with few rows use a smaller, more regularised space.
  4. Re-runs CV with the best parameters and a fixed number of rounds to get clean
     out-of-fold (OOF) predictions, and compares them with a logistic-regression baseline.
  5. Fits the final model on all training rows, saves model, importances (gain and mean |SHAP|),
     OOF predictions and metrics to models/<SEASON>/.
  6. Only with EVAL_TEST=1: scores the held-out test rows once, with block-bootstrap CIs.

Usage:
    SEASON=summer python train_lgbm.py
    SEASON=winter N_TRIALS=40 python train_lgbm.py
    SEASON=winter EVAL_TEST=1 python train_lgbm.py      # only when you are done tuning

Requires:  pip install lightgbm scikit-learn pandas pyarrow scipy
and make_split.py in the same folder (for cv_splits).
"""

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from make_split import cv_splits

# ------------------------------- settings --------------------------------------
ROOT = Path(__file__).resolve().parent
PROC = Path(os.environ.get("PROC_DIR", ROOT / "data/processed"))
MODEL_DIR = PROC / "panel_same_date_model"
SPLIT_PATH = PROC / "split.parquet"

SEASON = os.environ.get("SEASON", "summer")             # summer | winter | pooled
N_TRIALS = int(os.environ.get("N_TRIALS", "25"))
CV_BUFFER_KM = float(os.environ.get("CV_BUFFER_KM", "10"))
EVAL_TEST = os.environ.get("EVAL_TEST", "0") == "1"
USE_COORDS = os.environ.get("USE_COORDS", "0") == "1"   # add x_km, y_km as features (location proxy!)
SCALE_POS_WEIGHT = float(os.environ.get("SCALE_POS_WEIGHT", "1.0"))
SEED = int(os.environ.get("SEED", "0"))
SMALL_N = 5000          # fewer training rows than this -> constrained parameter space
MAX_ROUNDS = 3000
EARLY_STOP = 100
N_BOOT = 500

OUT_DIR = ROOT / "models" / SEASON
LABEL = "fire"
META = {"date", "pixel_id", "year", "season", "season_year", "season_id", "season_complete",
        "lc_year", "role", "cv_fold", "block_cv", "block_test", "x_km", "y_km"}
NOT_FEATURES = {"lc_sum", "valid_frac", "ndvi_n_images"}      # quality flags
EXCLUDE = {"elevation", "dem_land_frac", "elev_std"}          # same choices as the exploration
LC_GROUPS = {
    "lcg_forest": ["evergreen_needle", "evergreen_broad", "decid_needle", "decid_broad", "mixed_forest"],
    "lcg_shrub": ["closed_shrub", "open_shrub"],
    "lcg_savanna": ["woody_savanna", "savanna"],
    "lcg_grass": ["grassland"],
    "lcg_crop": ["cropland", "crop_mosaic"],
    "lcg_wetland": ["wetland"],
    "lcg_urban": ["urban"],
    "lcg_barren_snow": ["barren", "snow_ice"],
    "lcg_water": ["water"],
}
# --------------------------------------------------------------------------------

BASE_PARAMS = {
    "objective": "binary", "metric": "auc", "verbosity": -1, "seed": SEED,
    "bagging_seed": SEED, "feature_fraction_seed": SEED, "deterministic": True,
    "force_row_wise": True, "scale_pos_weight": SCALE_POS_WEIGHT,
}
DEFAULTS_SMALL = {"learning_rate": 0.03, "num_leaves": 9, "max_depth": 4, "min_child_samples": 40,
                  "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                  "lambda_l1": 0.1, "lambda_l2": 10.0}
DEFAULTS_LARGE = {"learning_rate": 0.05, "num_leaves": 31, "max_depth": -1, "min_child_samples": 50,
                  "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                  "lambda_l1": 0.0, "lambda_l2": 1.0}


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_params(rng, small: bool) -> dict:
    if small:   # few rows: shallow trees, big leaves, strong regularisation
        return {"learning_rate": 0.03, "num_leaves": int(rng.integers(4, 16)),
                "max_depth": int(rng.choice([3, 4, 5])), "min_child_samples": int(rng.integers(20, 61)),
                "feature_fraction": float(rng.uniform(0.5, 0.9)), "bagging_fraction": float(rng.uniform(0.6, 0.9)),
                "bagging_freq": 1, "lambda_l1": loguniform(rng, 1e-2, 10), "lambda_l2": loguniform(rng, 1, 50)}
    return {"learning_rate": 0.05, "num_leaves": int(rng.integers(15, 128)),
            "max_depth": int(rng.choice([-1, 6, 8, 10])), "min_child_samples": int(rng.integers(20, 201)),
            "feature_fraction": float(rng.uniform(0.5, 1.0)), "bagging_fraction": float(rng.uniform(0.6, 1.0)),
            "bagging_freq": 1, "lambda_l1": loguniform(rng, 1e-3, 10), "lambda_l2": loguniform(rng, 1e-3, 30)}


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    base = float(y.mean())
    out = {"n": int(len(y)), "fires": int(y.sum()), "base_rate": base}
    if y.min() == y.max():
        return out
    k = max(1, int(0.1 * len(y)))
    top = np.argsort(-p)[:k]
    out.update({
        "auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "pr_auc_lift": average_precision_score(y, p) / base,
        "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "brier": brier_score_loss(y, p),
        "precision_top10pct": float(y[top].mean()), "lift_top10pct": float(y[top].mean() / base),
    })
    return out


def block_bootstrap(y, p, blocks, n=N_BOOT, seed=0) -> dict:
    """95% CI for AUC / PR-AUC by resampling whole spatial blocks."""
    rng = np.random.default_rng(seed)
    codes = pd.factorize(blocks)[0]
    idx_by = [np.flatnonzero(codes == b) for b in range(codes.max() + 1)]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(idx_by), len(idx_by))
        ii = np.concatenate([idx_by[b] for b in pick])
        if y[ii].min() == y[ii].max():
            continue
        vals.append((roc_auc_score(y[ii], p[ii]), average_precision_score(y[ii], p[ii])))
    arr = np.array(vals)
    return {"auc_ci95": np.percentile(arr[:, 0], [2.5, 97.5]), "pr_auc_ci95": np.percentile(arr[:, 1], [2.5, 97.5]),
            "n_blocks": len(idx_by), "n_boot_ok": len(arr)}


# ------------------------------- data ------------------------------------------
def load_data():
    df = pd.read_parquet(MODEL_DIR)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    split = pd.read_parquet(SPLIT_PATH, columns=["pixel_id", "date", "role", "cv_fold", "block_cv", "x_km", "y_km"])
    df = df.merge(split, on=["pixel_id", "date"], validate="1:1")

    raw_lc = [c for c in df.columns if c.startswith("lc_") and c not in ("lc_sum", "lc_year")]
    made = []
    for g, names in LC_GROUPS.items():
        cols = [f"lc_{n}" for n in names if f"lc_{n}" in df.columns]
        if cols:
            df[g] = df[cols].sum(axis=1)
            made.append(g)
    excl = set(EXCLUDE) | (set(raw_lc) if made else set(made))
    return df, excl


def run_cv(params, X, y, folds, rounds=None):
    """Spatial CV. rounds=None -> early stopping on the validation fold; else fixed rounds."""
    oof = np.full(len(y), np.nan)
    best, aucs = [], []
    for fit_idx, val_idx in folds:
        dtr = lgb.Dataset(X.iloc[fit_idx], y[fit_idx])
        if rounds is None:
            dva = lgb.Dataset(X.iloc[val_idx], y[val_idx], reference=dtr)
            m = lgb.train(params, dtr, MAX_ROUNDS, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
            bi = m.best_iteration or m.current_iteration()
        else:
            m = lgb.train(params, dtr, rounds)
            bi = rounds
        oof[val_idx] = m.predict(X.iloc[val_idx], num_iteration=bi)
        best.append(bi)
        aucs.append(roc_auc_score(y[val_idx], oof[val_idx]) if len(set(y[val_idx])) > 1 else np.nan)
    return oof, best, aucs


def run_baseline(X, y, folds):
    oof = np.full(len(y), np.nan)
    for fit_idx, val_idx in folds:
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=2000))
        pipe.fit(X.iloc[fit_idx], y[fit_idx])
        oof[val_idx] = pipe.predict_proba(X.iloc[val_idx])[:, 1]
    return oof


def group_metrics(frame, p, with_ci=False, boot_blocks=None):
    """Metrics overall, on complete seasons, and (pooled) per season."""
    y = frame[LABEL].values
    groups = {"all": np.ones(len(frame), bool), "complete_seasons": frame["season_complete"].astype(bool).values}
    if SEASON == "pooled":
        for s in ("summer", "winter"):
            groups[s] = (frame["season"] == s).values
    res = {}
    for name, m in groups.items():
        if m.sum() == 0:
            continue
        res[name] = metrics(y[m], p[m])
        if with_ci and boot_blocks is not None and len(set(y[m])) > 1:
            res[name].update(block_bootstrap(y[m], p[m], boot_blocks[m]))
    return res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, excl = load_data()

    sub = df if SEASON == "pooled" else df[df["season"] == SEASON]
    if SEASON not in ("summer", "winter", "pooled"):
        raise SystemExit("SEASON must be summer, winter or pooled")
    sub = sub.copy()
    if SEASON == "pooled":
        sub["is_summer"] = (sub["season"] == "summer").astype(int)
    train = sub[sub["role"] == "train"].reset_index(drop=True)
    test = sub[sub["role"] == "test"].reset_index(drop=True)

    skip = META | NOT_FEATURES | excl | {LABEL}
    feats = [c for c in train.columns
             if c not in skip and pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique() > 1]
    if USE_COORDS:
        feats += ["x_km", "y_km"]
    X, y = train[feats], train[LABEL].values.astype(int)
    small = len(train) < SMALL_N
    print(f"{SEASON}: {len(train):,} train rows ({y.sum():,} fires, {y.mean():.1%}) | "
          f"{len(test):,} test rows (not used unless EVAL_TEST=1) | {len(feats)} features | "
          f"{'small' if small else 'large'} search space")
    print("features:", feats)

    folds = list(cv_splits(train, buffer_km=CV_BUFFER_KM))
    fires_per_fold = [int(y[v].sum()) for _, v in folds]
    print(f"{len(folds)} spatial folds, buffer {CV_BUFFER_KM} km, fires per fold: {fires_per_fold}")
    if min(fires_per_fold) < 30:
        print("WARNING: a fold has fewer than 30 fires, so fold AUCs and the parameter search are very noisy. "
              "Use fewer folds or a simpler model, and treat CV differences below ~0.03 AUC as noise.")

    # ---- random search ----
    rng = np.random.default_rng(SEED)
    space = [DEFAULTS_SMALL if small else DEFAULTS_LARGE] + [sample_params(rng, small) for _ in range(N_TRIALS)]
    rows = []
    for t, sp in enumerate(space):
        _, best, aucs = run_cv({**BASE_PARAMS, **sp}, X, y, folds)
        rows.append({"trial": t, "cv_auc": float(np.nanmean(aucs)), "cv_auc_std": float(np.nanstd(aucs)),
                     "rounds": int(np.median(best)), **sp})
        print(f"trial {t:3d}  AUC {rows[-1]['cv_auc']:.4f} +- {rows[-1]['cv_auc_std']:.4f}  rounds {rows[-1]['rounds']}")
    trials = pd.DataFrame(rows).sort_values("cv_auc", ascending=False)
    trials.to_csv(OUT_DIR / "trials.csv", index=False)

    best_row = trials.iloc[0]
    keys = list(DEFAULTS_SMALL)
    params = {**BASE_PARAMS, **{k: best_row[k].item() if hasattr(best_row[k], "item") else best_row[k] for k in keys}}
    params["num_leaves"], params["max_depth"] = int(params["num_leaves"]), int(params["max_depth"])
    params["min_child_samples"], params["bagging_freq"] = int(params["min_child_samples"]), int(params["bagging_freq"])
    rounds = int(max(10, round(best_row["rounds"] * 1.1)))     # a bit more data in the final fit
    print(f"\nbest trial {int(best_row['trial'])}: AUC {best_row['cv_auc']:.4f} | {rounds} rounds")

    # ---- clean OOF predictions with fixed rounds, and the baseline ----
    oof, _, fold_aucs = run_cv(params, X, y, folds, rounds=rounds)
    base_oof = run_baseline(X, y, folds)
    covered = ~np.isnan(oof)
    cv_res = {"lightgbm": group_metrics(train[covered], oof[covered]),
              "logistic_baseline": group_metrics(train[covered], base_oof[covered]),
              "fold_auc": fold_aucs, "fold_auc_mean": float(np.nanmean(fold_aucs)),
              "fold_auc_std": float(np.nanstd(fold_aucs)),
              "note": "hyperparameters were chosen on these same folds, so CV scores are slightly optimistic"}
    (OUT_DIR / "cv_metrics.json").write_text(json.dumps(cv_res, indent=2, default=_np))

    print("\n--- spatial CV (out-of-fold) ---")
    print(f"fold AUCs: {np.round(fold_aucs, 3).tolist()}  mean {cv_res['fold_auc_mean']:.3f} +- {cv_res['fold_auc_std']:.3f}")
    table = pd.DataFrame({f"{m}/{g}": v for m in ("lightgbm", "logistic_baseline") for g, v in cv_res[m].items()}).T
    print(table[["n", "fires", "base_rate", "auc", "pr_auc", "pr_auc_lift", "logloss", "lift_top10pct"]].round(3).to_string())

    # ---- final model and importances ----
    final = lgb.train(params, lgb.Dataset(X, y), rounds)
    final.save_model(str(OUT_DIR / "model.txt"))
    imp = pd.DataFrame({"feature": final.feature_name(),
                        "gain": final.feature_importance("gain"), "splits": final.feature_importance("split")})
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    sample = X.sample(min(5000, len(X)), random_state=SEED)
    contrib = final.predict(sample, pred_contrib=True)[:, :-1]           # SHAP values from LightGBM itself
    imp["shap_mean_abs"] = np.abs(contrib).mean(axis=0)
    imp = imp.sort_values("shap_mean_abs", ascending=False)
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print("\ntop features by mean |SHAP|:")
    print(imp.head(10).round(4).to_string(index=False))

    pd.DataFrame({"pixel_id": train["pixel_id"], "date": train["date"], "season": train["season"],
                  "fire": y, "oof_pred": oof, "oof_baseline": base_oof, "cv_fold": train["cv_fold"]}
                 ).to_parquet(OUT_DIR / "oof_predictions.parquet", index=False)
    (OUT_DIR / "params.json").write_text(json.dumps(
        {"params": params, "rounds": rounds, "cv_buffer_km": CV_BUFFER_KM, "season": SEASON,
         "n_train": len(train), "n_trials": N_TRIALS, "seed": SEED}, indent=2, default=_np))
    (OUT_DIR / "features.json").write_text(json.dumps(feats, indent=2))

    # ---- test set: once, only on request ----
    if not EVAL_TEST:
        print("\nTest set NOT evaluated. When tuning is finished, run once with EVAL_TEST=1.")
        return
    p = final.predict(test[feats])
    res = group_metrics(test, p, with_ci=True, boot_blocks=test["block_cv"].values)
    (OUT_DIR / "test_metrics.json").write_text(json.dumps(res, indent=2, default=_np))
    pd.DataFrame({"pixel_id": test["pixel_id"], "date": test["date"], "season": test["season"],
                  "fire": test[LABEL], "pred": p}).to_parquet(OUT_DIR / "test_predictions.parquet", index=False)
    print("\n--- TEST (evaluate once; do not tune on this) ---")
    for g, v in res.items():
        ci = v.get("auc_ci95")
        ci_txt = f" AUC 95% CI [{ci[0]:.3f}, {ci[1]:.3f}] ({v['n_blocks']} blocks)" if ci is not None else ""
        print(f"{g:18s} n={v['n']:,} fires={v['fires']:,} base={v['base_rate']:.3f} "
              f"AUC={v.get('auc', float('nan')):.3f} PR-AUC={v.get('pr_auc', float('nan')):.3f}{ci_txt}")


if __name__ == "__main__":
    main()
