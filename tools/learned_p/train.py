#!/usr/bin/env python3
"""Train the tiny learned-p model with deterministic NumPy logistic/ridge regression."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


CATEGORICAL = ("shape", "colour", "hand", "target_bin")
NUMERIC = ("attempt_index", "pelvis_x", "pelvis_y", "distance", "nearest_neighbor_distance", "time_remaining")


def read_rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty dataset")
    return rows


def schema(rows):
    names = []
    for key in CATEGORICAL:
        values = sorted({(row.get(key) or "unknown") for row in rows} | {"unknown"})
        names.extend(f"{key}={value}" for value in values)
    names.extend(NUMERIC)
    return names


def raw_matrix(rows, names):
    x = np.empty((len(rows), len(names)), dtype=float)
    for j, name in enumerate(names):
        if "=" in name:
            key, value = name.split("=", 1)
            x[:, j] = [float((row.get(key) or "unknown") == value) for row in rows]
        else:
            vals = []
            for row in rows:
                try:
                    vals.append(float(row[name]))
                except (KeyError, TypeError, ValueError):
                    vals.append(np.nan)
            x[:, j] = vals
    return x


def standardize_fit(x):
    valid = np.isfinite(x)
    counts = valid.sum(axis=0)
    means = np.divide(np.where(valid, x, 0.0).sum(axis=0), counts, out=np.zeros(x.shape[1]), where=counts > 0)
    filled = np.where(np.isnan(x), means, x)
    stds = np.std(filled, axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return means, stds


def transform(x, means, stds):
    return (np.where(np.isnan(x), means, x) - means) / stds


def sigmoid(z):
    z = np.clip(np.asarray(z, float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(x, y, l2=1.0, iterations=80):
    n, d = x.shape
    theta = np.zeros(d + 1, dtype=float)
    design = np.column_stack((np.ones(n), x))
    reg = np.r_[0.0, np.full(d, float(l2))]
    for _ in range(iterations):
        p = sigmoid(design @ theta)
        grad = design.T @ (p - y) / n + reg * theta
        curvature = np.maximum(p * (1.0 - p), 1e-6)
        hessian = (design.T * curvature) @ design / n + np.diag(reg)
        step = np.linalg.solve(hessian + np.eye(d + 1) * 1e-10, grad)
        theta -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return theta[1:], float(theta[0])


def fit_platt(scores, y):
    # P(y=1|score) = sigmoid(a*score+b); weak regularisation stabilises small datasets.
    w, b = fit_logistic(np.asarray(scores, float)[:, None], y, l2=1e-4, iterations=80)
    return float(w[0]), float(b)


def fit_duration(x, duration, l2=1.0):
    design = np.column_stack((np.ones(len(x)), x))
    reg = np.eye(design.shape[1]) * float(l2)
    reg[0, 0] = 0.0
    theta = np.linalg.solve(design.T @ design + reg, design.T @ duration)
    return theta[1:], float(theta[0])


def auc(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pos, neg = int(y.sum()), int((1 - y).sum())
    if not pos or not neg:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and p[order[end]] == p[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def folds_by_episode(rows, n_folds=5):
    episodes = sorted({row["episode"] for row in rows})
    if len(episodes) < n_folds:
        raise ValueError(f"need at least {n_folds} episodes, got {len(episodes)}")
    fold_of = {episode: i % n_folds for i, episode in enumerate(episodes)}
    return np.array([fold_of[row["episode"]] for row in rows], int)


def train(rows, l2=1.0, duration_l2=1.0, n_folds=5):
    names = schema(rows)
    raw = raw_matrix(rows, names)
    y = np.array([int(row["label"]) for row in rows], float)
    duration = np.array([float(row["duration_s"]) for row in rows], float)
    prior = np.clip(np.array([float(row["p_prior"]) for row in rows], float), 0.05, 0.98)
    fold = folds_by_episode(rows, n_folds)
    oof_score = np.zeros(len(rows), float)
    oof_p = np.zeros(len(rows), float)
    oof_duration = np.zeros(len(rows), float)
    for k in range(n_folds):
        train_idx, test_idx = fold != k, fold == k
        means, stds = standardize_fit(raw[train_idx])
        x_train, x_test = transform(raw[train_idx], means, stds), transform(raw[test_idx], means, stds)
        weights, bias = fit_logistic(x_train, y[train_idx], l2=l2)
        oof_score[test_idx] = bias + x_test @ weights
        dw, db = fit_duration(x_train, duration[train_idx], duration_l2)
        oof_duration[test_idx] = db + x_test @ dw
        # Calibrate this outer fold using only its training episodes.  Fitting Platt on
        # every OOF score and reporting those same rows would make Brier optimistic.
        outer_train = np.flatnonzero(train_idx)
        inner_rows = [rows[i] for i in outer_train]
        inner_folds = min(n_folds, len({row["episode"] for row in inner_rows}))
        inner_fold = folds_by_episode(inner_rows, inner_folds)
        inner_score = np.zeros(len(outer_train), float)
        for inner_k in range(inner_folds):
            inner_train, inner_test = inner_fold != inner_k, inner_fold == inner_k
            inner_raw = raw[outer_train]
            inner_means, inner_stds = standardize_fit(inner_raw[inner_train])
            inner_x_train = transform(inner_raw[inner_train], inner_means, inner_stds)
            inner_x_test = transform(inner_raw[inner_test], inner_means, inner_stds)
            inner_weights, inner_bias = fit_logistic(inner_x_train, y[outer_train][inner_train], l2=l2)
            inner_score[inner_test] = inner_bias + inner_x_test @ inner_weights
        fold_a, fold_b = fit_platt(inner_score, y[outer_train])
        oof_p[test_idx] = sigmoid(fold_a * oof_score[test_idx] + fold_b)
    # The exported calibrator can use all cross-fitted scores; final model weights are
    # subsequently trained on all rows.  Reported metrics use nested calibration above.
    a, b = fit_platt(oof_score, y)
    logged_prior = np.array([row.get("p_prior_source") in ("confidence_order", "learned_log") for row in rows])
    metrics = {
        "rows": len(rows), "episodes": len({row["episode"] for row in rows}), "folds": n_folds,
        "success_rate": float(y.mean()), "model_auc": auc(y, oof_p), "model_brier": brier(y, oof_p),
        "prior_auc": auc(y, prior), "prior_brier": brier(y, prior),
        "duration_rmse_s": float(math.sqrt(np.mean((oof_duration - duration) ** 2))),
    }
    if logged_prior.any():
        metrics.update({
            "logged_prior_rows": int(logged_prior.sum()),
            "logged_model_auc": auc(y[logged_prior], oof_p[logged_prior]),
            "logged_model_brier": brier(y[logged_prior], oof_p[logged_prior]),
            "logged_prior_auc": auc(y[logged_prior], prior[logged_prior]),
            "logged_prior_brier": brier(y[logged_prior], prior[logged_prior]),
        })
    means, stds = standardize_fit(raw)
    x = transform(raw, means, stds)
    weights, bias = fit_logistic(x, y, l2=l2)
    duration_weights, duration_bias = fit_duration(x, duration, duration_l2)
    model = {
        "version": 1, "feature_names": names, "means": means.tolist(), "stds": stds.tolist(),
        "weights": weights.tolist(), "bias": bias, "calibration": {"a": a, "b": b},
        "duration": {"weights": duration_weights.tolist(), "bias": duration_bias},
        "training": metrics,
    }
    return model, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="CSV from build_dataset.py")
    parser.add_argument("-o", "--output", required=True, help="output model JSON")
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--duration-l2", type=float, default=1.0)
    args = parser.parse_args(argv)
    model, metrics = train(read_rows(args.dataset), args.l2, args.duration_l2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    print(f"rows={metrics['rows']} episodes={metrics['episodes']} 5-fold-by-episode")
    print(f"model  AUC={metrics['model_auc']:.4f} Brier={metrics['model_brier']:.4f}")
    print(f"prior  AUC={metrics['prior_auc']:.4f} Brier={metrics['prior_brier']:.4f}")
    if metrics.get("logged_prior_rows"):
        print(f"logged-prior comparison ({metrics['logged_prior_rows']} rows): "
              f"model AUC={metrics['logged_model_auc']:.4f} Brier={metrics['logged_model_brier']:.4f}; "
              f"prior AUC={metrics['logged_prior_auc']:.4f} Brier={metrics['logged_prior_brier']:.4f}")
    print(f"duration RMSE={metrics['duration_rmse_s']:.3f}s")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
