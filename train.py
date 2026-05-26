from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


RANDOM_SEED = 42
MODEL_PATH = Path("model.pkl")
FALLBACK_MODEL_PATH = Path("model_fallback.npz")
DEFAULT_DATA_CANDIDATES = ("DH_FR1.mat",)
KRR_GAMMA_FACTOR = 0.5
KRR_LAMBDA = 0.1


def load_project_data(path: str | Path | None = None, require_p: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path is None:
        for candidate in DEFAULT_DATA_CANDIDATES:
            if Path(candidate).exists():
                path = candidate
                break
    if path is None:
        raise FileNotFoundError("DH_FR1.mat 파일을 찾지 못했습니다.")

    data = sio.loadmat(path, squeeze_me=False)
    if "d_hat" not in data:
        raise KeyError("입력 .mat 파일에 d_hat 변수가 없습니다.")
    if "p_bs" in data:
        p_bs = np.asarray(data["p_bs"], dtype=float)
    elif "BS_positions" in data:
        p_bs = np.asarray(data["BS_positions"], dtype=float)
    else:
        raise KeyError("입력 .mat 파일에 p_bs 또는 BS_positions 변수가 없습니다.")

    d_hat = np.asarray(data["d_hat"], dtype=float)
    p = np.asarray(data["p"], dtype=float) if "p" in data else None
    if require_p and p is None:
        raise KeyError("train.py 학습에는 GT p 변수가 필요합니다.")

    p_bs = ensure_shape(p_bs, rows=2, name="p_bs")
    d_hat = ensure_shape(d_hat, rows=p_bs.shape[1], name="d_hat")
    if p is not None:
        p = ensure_shape(p, rows=2, name="p")
    return p, d_hat, p_bs


def ensure_shape(arr: np.ndarray, rows: int, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    if arr.shape[0] == rows:
        return arr
    if arr.shape[1] == rows:
        return arr.T
    raise ValueError(f"{name} has incompatible shape {arr.shape}; expected one axis to be {rows}")


def true_ranges(p: np.ndarray, p_bs: np.ndarray) -> np.ndarray:
    user_xy = p.T[:, None, :]
    anchor_xy = p_bs.T[None, :, :]
    return np.linalg.norm(user_xy - anchor_xy, axis=2).T


def fit_affine_calibration(d_hat: np.ndarray, d_true: np.ndarray) -> dict[str, np.ndarray]:
    num_anchor = d_hat.shape[0]
    scale = np.ones(num_anchor)
    offset = np.zeros(num_anchor)
    sigma = np.ones(num_anchor)

    for i in range(num_anchor):
        x = d_hat[i]
        y = d_true[i]
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y >= 0)
        if mask.sum() < 10:
            sigma[i] = 5.0
            continue

        x_fit = x[mask]
        y_fit = y[mask]
        q_lo, q_hi = np.percentile(x_fit, [1, 99])
        trim = (x_fit >= q_lo) & (x_fit <= q_hi)
        x_fit = x_fit[trim]
        y_fit = y_fit[trim]

        a, b = linear_fit(x_fit, y_fit)
        residual = y_fit - (a * x_fit + b)
        med = np.median(residual)
        mad = np.median(np.abs(residual - med)) + 1e-6
        keep = np.abs(residual - med) <= 3.5 * 1.4826 * mad
        if keep.sum() >= 10:
            a, b = linear_fit(x_fit[keep], y_fit[keep])
            residual = y_fit[keep] - (a * x_fit[keep] + b)

        scale[i] = float(np.clip(a, 0.02, 5.0))
        offset[i] = float(b)
        sigma[i] = float(np.clip(1.4826 * np.median(np.abs(residual - np.median(residual))) + 0.5, 0.75, 30.0))

    return {"scale": scale, "offset": offset, "sigma": sigma}


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def apply_calibration(d_hat: np.ndarray, calibration: dict[str, np.ndarray], max_range: float) -> np.ndarray:
    ranges = calibration["scale"][:, None] * d_hat + calibration["offset"][:, None]
    ranges = np.where(np.isfinite(ranges), ranges, max_range)
    return np.clip(ranges, 0.05, max_range)


def estimate_bounds(p: np.ndarray, p_bs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_points = np.column_stack([p, p_bs])
    lo = np.nanmin(all_points, axis=1) - 8.0
    hi = np.nanmax(all_points, axis=1) + 8.0
    return lo.astype(float), hi.astype(float)


def weighted_centroid(ranges: np.ndarray, p_bs: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    valid = np.isfinite(ranges) & (ranges > 0)
    if valid.sum() == 0:
        return (bounds[0] + bounds[1]) / 2.0
    weights = 1.0 / np.maximum(ranges[valid], 0.5) ** 2
    pos = (p_bs[:, valid] * weights[None, :]).sum(axis=1) / weights.sum()
    return np.clip(pos, bounds[0], bounds[1])


def linear_initial(ranges: np.ndarray, p_bs: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    valid = np.flatnonzero(np.isfinite(ranges) & (ranges > 0))
    if len(valid) < 3:
        return weighted_centroid(ranges, p_bs, bounds)

    ref = valid[np.argmin(ranges[valid])]
    rows = []
    rhs = []
    weights = []
    a0 = p_bs[:, ref]
    r0 = ranges[ref]
    for j in valid:
        if j == ref:
            continue
        aj = p_bs[:, j]
        rows.append(2.0 * (aj - a0))
        rhs.append(r0 * r0 - ranges[j] * ranges[j] + np.dot(aj, aj) - np.dot(a0, a0))
        weights.append(1.0 / max(ranges[j], 0.5))
    if len(rows) < 2:
        return weighted_centroid(ranges, p_bs, bounds)

    a = np.asarray(rows, dtype=float)
    b = np.asarray(rhs, dtype=float)
    w = np.sqrt(np.asarray(weights, dtype=float))
    try:
        x, *_ = np.linalg.lstsq(a * w[:, None], b * w, rcond=None)
        if np.all(np.isfinite(x)):
            return np.clip(x, bounds[0], bounds[1])
    except np.linalg.LinAlgError:
        pass
    return weighted_centroid(ranges, p_bs, bounds)


def robust_multilateration(
    ranges: np.ndarray,
    p_bs: np.ndarray,
    sigma: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    valid = np.isfinite(ranges) & (ranges > 0)
    if valid.sum() < 3:
        return weighted_centroid(ranges, p_bs, bounds)

    anchors = p_bs[:, valid].T
    d = ranges[valid]
    s = np.clip(sigma[valid], 0.75, 30.0)
    x0 = linear_initial(ranges, p_bs, bounds)

    def residual(x: np.ndarray) -> np.ndarray:
        return (np.linalg.norm(anchors - x[None, :], axis=1) - d) / s

    try:
        result = least_squares(
            residual,
            x0,
            bounds=(bounds[0], bounds[1]),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=80,
            xtol=1e-5,
            ftol=1e-5,
            gtol=1e-5,
        )
        if result.success and np.all(np.isfinite(result.x)):
            return result.x.astype(float)
    except Exception:
        pass
    return x0.astype(float)


def batch_geo_positions(
    ranges: np.ndarray,
    p_bs: np.ndarray,
    sigma: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    p_geo = np.zeros((2, ranges.shape[1]), dtype=float)
    for idx in range(ranges.shape[1]):
        p_geo[:, idx] = robust_multilateration(ranges[:, idx], p_bs, sigma, bounds)
    return p_geo


def geometry_condition(p_geo: np.ndarray, p_bs: np.ndarray) -> float:
    diff = p_geo[:, None] - p_bs
    dist = np.linalg.norm(diff, axis=0)
    mask = dist > 1e-6
    if mask.sum() < 2:
        return 1e3
    jac = diff[:, mask].T / dist[mask, None]
    singular = np.linalg.svd(jac, compute_uv=False)
    if len(singular) < 2 or singular[-1] < 1e-6:
        return 1e3
    return float(np.clip(singular[0] / singular[-1], 1.0, 1e3))


def make_features(ranges: np.ndarray, p_geo: np.ndarray, p_bs: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    num_user = ranges.shape[1]
    feats: list[np.ndarray] = []
    for u in range(num_user):
        rg = ranges[:, u]
        geo = p_geo[:, u]
        calc = np.linalg.norm(p_bs - geo[:, None], axis=0)
        residual = calc - rg
        abs_res = np.abs(residual)
        norm_res = residual / np.clip(sigma, 0.75, 30.0)
        cond = geometry_condition(geo, p_bs)

        q = np.percentile(rg, [10, 25, 50, 75, 90])
        rq = np.percentile(residual, [10, 25, 50, 75, 90])
        arq = np.percentile(abs_res, [50, 75, 90, 95])
        stats = np.array(
            [
                rg.mean(),
                rg.std(),
                rg.min(),
                rg.max(),
                residual.mean(),
                residual.std(),
                residual.min(),
                residual.max(),
                abs_res.mean(),
                abs_res.std(),
                abs_res.max(),
                np.linalg.norm(residual),
                cond,
            ],
            dtype=float,
        )
        feats.append(
            np.concatenate(
                [
                    geo,
                    rg,
                    np.sort(rg),
                    calc,
                    residual,
                    abs_res,
                    norm_res,
                    q,
                    rq,
                    arq,
                    stats,
                ]
            )
        )
    return np.asarray(feats, dtype=float)


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = np.linalg.norm(pred - target, axis=0)
    return {
        "MAE": float(np.mean(err)),
        "RMSE": float(np.sqrt(np.mean(err * err))),
        "Median": float(np.median(err)),
        "P90": float(np.percentile(err, 90)),
        "Max": float(np.max(err)),
    }


def print_metrics(label: str, values: dict[str, float]) -> None:
    items = ", ".join(f"{k}={v:.4f}" for k, v in values.items())
    print(f"{label}: {items}")


def make_residual_models(seed: int) -> list:
    return [
        MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=16,
                l2_regularization=0.1,
                random_state=seed,
            )
        ),
        ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=seed,
            n_jobs=-1,
        ),
    ]


def fit_residual_models(features: np.ndarray, target_delta: np.ndarray, seed: int) -> list:
    models = make_residual_models(seed)
    for model in models:
        model.fit(features, target_delta)
    return models


def predict_delta(models: list, features: np.ndarray) -> np.ndarray:
    predictions = [model.predict(features) for model in models]
    return np.mean(predictions, axis=0)


def fit_kernel_ridge(features: np.ndarray, target_delta: np.ndarray) -> dict[str, np.ndarray | float]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0) + 1e-6
    x_train = (features - mean) / scale
    gamma = KRR_GAMMA_FACTOR / x_train.shape[1]
    diff = x_train[:, None, :] - x_train[None, :, :]
    kernel = np.exp(-gamma * np.sum(diff * diff, axis=2))
    alpha = np.linalg.solve(kernel + KRR_LAMBDA * np.eye(kernel.shape[0]), target_delta)
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "feature_train": x_train,
        "alpha": alpha,
        "gamma": float(gamma),
        "lambda": float(KRR_LAMBDA),
    }


def predict_kernel_ridge(model: dict[str, np.ndarray | float], features: np.ndarray) -> np.ndarray:
    x_test = (features - model["feature_mean"]) / model["feature_scale"]
    diff = x_test[:, None, :] - model["feature_train"][None, :, :]
    kernel = np.exp(-float(model["gamma"]) * np.sum(diff * diff, axis=2))
    return kernel @ model["alpha"]


def save_fallback_model(path: Path, final_pack: dict, p_bs: np.ndarray, krr: dict[str, np.ndarray | float]) -> None:
    np.savez_compressed(
        path,
        scale=final_pack["calibration"]["scale"],
        offset=final_pack["calibration"]["offset"],
        sigma=final_pack["calibration"]["sigma"],
        bounds_lo=final_pack["bounds"][0],
        bounds_hi=final_pack["bounds"][1],
        max_range=np.array(final_pack["max_range"], dtype=float),
        p_bs_train=p_bs,
        feature_mean=krr["feature_mean"],
        feature_scale=krr["feature_scale"],
        feature_train=krr["feature_train"],
        alpha=krr["alpha"],
        gamma=np.array(krr["gamma"], dtype=float),
        krr_lambda=np.array(krr["lambda"], dtype=float),
    )


def train_once(p: np.ndarray, d_hat: np.ndarray, p_bs: np.ndarray, indices: np.ndarray) -> dict:
    p_fit = p[:, indices]
    d_true = true_ranges(p_fit, p_bs)
    calibration = fit_affine_calibration(d_hat[:, indices], d_true)
    max_range = float(np.linalg.norm(np.ptp(np.column_stack([p_fit, p_bs]), axis=1)) * 1.8 + 20.0)
    bounds = estimate_bounds(p_fit, p_bs)
    ranges = apply_calibration(d_hat, calibration, max_range=max_range)
    p_geo = batch_geo_positions(ranges[:, indices], p_bs, calibration["sigma"], bounds)
    features = make_features(ranges[:, indices], p_geo, p_bs, calibration["sigma"])
    target_delta = (p[:, indices] - p_geo).T
    return {
        "calibration": calibration,
        "bounds": bounds,
        "max_range": max_range,
        "features": features,
        "target_delta": target_delta,
        "p_geo": p_geo,
    }


def main() -> None:
    start = time.time()
    p, d_hat, p_bs = load_project_data()
    num_user = d_hat.shape[1]
    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(num_user)
    split = int(num_user * 0.8)
    train_idx = order[:split]
    valid_idx = order[split:]

    train_pack = train_once(p, d_hat, p_bs, train_idx)
    residual_models = fit_residual_models(train_pack["features"], train_pack["target_delta"], RANDOM_SEED)

    valid_ranges = apply_calibration(d_hat[:, valid_idx], train_pack["calibration"], train_pack["max_range"])
    valid_geo = batch_geo_positions(valid_ranges, p_bs, train_pack["calibration"]["sigma"], train_pack["bounds"])
    valid_features = make_features(valid_ranges, valid_geo, p_bs, train_pack["calibration"]["sigma"])
    valid_delta = predict_delta(residual_models, valid_features).T
    valid_pred = valid_geo + valid_delta

    print_metrics("Validation robust geometry", metrics(valid_geo, p[:, valid_idx]))
    print_metrics("Validation residual correction", metrics(valid_pred, p[:, valid_idx]))

    final_indices = np.arange(num_user)
    final_pack = train_once(p, d_hat, p_bs, final_indices)
    final_models = fit_residual_models(final_pack["features"], final_pack["target_delta"], RANDOM_SEED)

    model = {
        "algorithm": "Geometry-Aware Residual Correction",
        "calibration": final_pack["calibration"],
        "bounds": final_pack["bounds"],
        "max_range": final_pack["max_range"],
        "p_bs_train": p_bs,
        "regressors": final_models,
        "feature_count": int(final_pack["features"].shape[1]),
        "random_seed": RANDOM_SEED,
    }
    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)

    train_geo_pred = final_pack["p_geo"] + predict_delta(final_models, final_pack["features"]).T
    print_metrics("Full-data fitted residual correction", metrics(train_geo_pred, p))
    fallback_model = fit_kernel_ridge(final_pack["features"], final_pack["target_delta"])
    save_fallback_model(FALLBACK_MODEL_PATH, final_pack, p_bs, fallback_model)

    fallback_delta = predict_kernel_ridge(fallback_model, final_pack["features"]).T
    fallback_pred = final_pack["p_geo"] + fallback_delta
    print_metrics("Full-data fitted fallback KRR", metrics(fallback_pred, p))
    print(f"Saved {MODEL_PATH} and {FALLBACK_MODEL_PATH} in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
