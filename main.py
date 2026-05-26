from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares


MODEL_PATH = Path("model.pkl")
FALLBACK_MODEL_PATH = Path("model_fallback.npz")
DEFAULT_DATA_CANDIDATES = ("DH_FR1.mat",)


def load_project_data(path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
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

    p_bs = ensure_shape(p_bs, rows=2, name="p_bs")
    d_hat = ensure_shape(np.asarray(data["d_hat"], dtype=float), rows=p_bs.shape[1], name="d_hat")
    return d_hat, p_bs


def ensure_shape(arr: np.ndarray, rows: int, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    if arr.shape[0] == rows:
        return arr
    if arr.shape[1] == rows:
        return arr.T
    raise ValueError(f"{name} has incompatible shape {arr.shape}; expected one axis to be {rows}")


def apply_calibration(d_hat: np.ndarray, calibration: dict[str, np.ndarray], max_range: float) -> np.ndarray:
    ranges = calibration["scale"][:, None] * d_hat + calibration["offset"][:, None]
    ranges = np.where(np.isfinite(ranges), ranges, max_range)
    return np.clip(ranges, 0.05, max_range)


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


def predict_delta(model: dict, features: np.ndarray) -> np.ndarray:
    if "regressors" in model:
        predictions = [regressor.predict(features) for regressor in model["regressors"]]
        return np.mean(predictions, axis=0)
    return model["regressor"].predict(features)


def predict_with_pickle_model(model: dict, d_hat: np.ndarray, p_bs: np.ndarray) -> np.ndarray:
    calibration = model["calibration"]
    bounds = model["bounds"]
    max_range = float(model["max_range"])
    ranges = apply_calibration(d_hat, calibration, max_range=max_range)
    p_geo = batch_geo_positions(ranges, p_bs, calibration["sigma"], bounds)
    features = make_features(ranges, p_geo, p_bs, calibration["sigma"])
    delta = predict_delta(model, features).T
    p_hat = p_geo + delta
    return np.asarray(p_hat, dtype=float)


def predict_with_fallback_model(path: Path, d_hat: np.ndarray, p_bs: np.ndarray) -> np.ndarray:
    data = np.load(path)
    calibration = {
        "scale": data["scale"],
        "offset": data["offset"],
        "sigma": data["sigma"],
    }
    bounds = (data["bounds_lo"], data["bounds_hi"])
    max_range = float(data["max_range"])
    ranges = apply_calibration(d_hat, calibration, max_range=max_range)
    p_geo = batch_geo_positions(ranges, p_bs, calibration["sigma"], bounds)
    features = make_features(ranges, p_geo, p_bs, calibration["sigma"])

    x_test = (features - data["feature_mean"]) / data["feature_scale"]
    diff = x_test[:, None, :] - data["feature_train"][None, :, :]
    kernel = np.exp(-float(data["gamma"]) * np.sum(diff * diff, axis=2))
    delta = (kernel @ data["alpha"]).T
    return np.asarray(p_geo + delta, dtype=float)


def main() -> np.ndarray:
    d_hat, p_bs = load_project_data()
    primary_error = None
    if MODEL_PATH.exists():
        try:
            with MODEL_PATH.open("rb") as f:
                model = pickle.load(f)
            return predict_with_pickle_model(model, d_hat, p_bs)
        except Exception as exc:
            primary_error = exc

    if FALLBACK_MODEL_PATH.exists():
        return predict_with_fallback_model(FALLBACK_MODEL_PATH, d_hat, p_bs)

    if primary_error is not None:
        raise primary_error
    raise FileNotFoundError("model.pkl 또는 model_fallback.npz 파일이 없습니다. 먼저 train.py를 실행해야 합니다.")


if __name__ == "__main__":
    pred = main()
    print(pred.shape)
