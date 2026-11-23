#!/usr/bin/env python3
"""Reproducible champion: minimal Joules_to_Q_min on this Intel CPU/RAPL setup.

Confirmed champions (Numba JIT, float32, backprop, one-cycle lr, early stop):
  - mnist12k, Q=0.94: 784-64-10,  batch 64,  max_lr 0.08, epochs 40  (~30-34 J median, 5/5 on split 42/123)
  - mnist-full, Q=0.95: 784-128-10, batch 128, max_lr 0.08, epochs 20 (~264 J median, 5/5 on standard split)

Metric: Joules_to_Q_min from Intel RAPL package-0 (energy_uj). JIT warmup is
performed before measurement and excluded. RAPL measures package energy, so
run on a quiet system (no parallel heavy workloads) for comparable numbers.

Usage:
  uv run --with numpy --with scikit-learn --with numba -- python run_champion.py
  uv run --with numpy --with scikit-learn --with numba -- python run_champion.py --task mnist-full --q-min 0.95
"""
from __future__ import annotations

import argparse, json, math, time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numba import njit


RAPL_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"


def read_rapl_uj(path=RAPL_PATH):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def load_mnist12k(split_seed=42, n_total=12000):
    d = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = d.data.astype(np.float32)
    y = d.target.astype(np.int64)
    X, _, y, _ = train_test_split(X, y, train_size=n_total, random_state=split_seed, stratify=y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=split_seed, stratify=y)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    return Xtr, Xte, ytr, yte, {"task": "mnist12k", "n_total": n_total, "split_seed": split_seed, "split": "stratified_80_20"}


def load_mnist_full():
    d = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = d.data.astype(np.float32)
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = X[:60000], X[60000:], y[:60000], y[60000:]
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    return Xtr, Xte, ytr, yte, {"task": "mnist-full", "n_train": 60000, "n_test": 10000, "split": "standard_60k_10k"}


@njit(cache=False)
def softmax_nb(z):
    out = np.empty_like(z)
    for i in range(z.shape[0]):
        m = z[i, 0]
        for j in range(1, z.shape[1]):
            if z[i, j] > m:
                m = z[i, j]
        s = 0.0
        for j in range(z.shape[1]):
            out[i, j] = math.exp(z[i, j] - m)
            s += out[i, j]
        invs = 1.0 / s
        for j in range(z.shape[1]):
            out[i, j] *= invs
    return out


@njit(cache=False)
def predict_nb(X, W1, b1, W2, b2):
    h = X @ W1
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h[i, j] + b1[j]
            h[i, j] = v if v > 0.0 else 0.0
    z = h @ W2
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            z[i, j] += b2[j]
    p = softmax_nb(z)
    pred = np.empty(X.shape[0], np.int64)
    for i in range(p.shape[0]):
        bi = 0
        bv = p[i, 0]
        for j in range(1, p.shape[1]):
            if p[i, j] > bv:
                bv = p[i, j]
                bi = j
        pred[i] = bi
    return pred


@njit(cache=False)
def train_epoch_nb_lrarr(X, y, W1, b1, W2, b2, lr_arr, batch, idx, step_offset):
    n = X.shape[0]
    step = step_offset
    h1 = W1.shape[1]
    out = W2.shape[1]
    for start in range(0, n, batch):
        end = start + batch
        if end > n:
            end = n
        k = end - start
        lr = lr_arr[step] if step < lr_arr.shape[0] else lr_arr[lr_arr.shape[0] - 1]
        step += 1
        xb = np.empty((k, X.shape[1]), np.float32)
        yb = np.empty(k, np.int64)
        for ii in range(k):
            src = idx[start + ii]
            yb[ii] = y[src]
            for j in range(X.shape[1]):
                xb[ii, j] = X[src, j]
        z1 = xb @ W1
        h = np.empty((k, h1), np.float32)
        for i in range(k):
            for j in range(h1):
                v = z1[i, j] + b1[j]
                h[i, j] = v if v > 0.0 else 0.0
        z2 = h @ W2
        for i in range(k):
            for j in range(out):
                z2[i, j] += b2[j]
        p = softmax_nb(z2)
        dz2 = p.copy()
        invk = np.float32(1.0 / k)
        for i in range(k):
            dz2[i, yb[i]] -= 1.0
            for j in range(out):
                dz2[i, j] *= invk
        gW2 = h.T @ dz2
        gb2 = np.zeros(out, np.float32)
        for i in range(k):
            for j in range(out):
                gb2[j] += dz2[i, j]
        dh = dz2 @ W2.T
        dz1 = np.empty((k, h1), np.float32)
        for i in range(k):
            for j in range(h1):
                dz1[i, j] = dh[i, j] if h[i, j] > 0.0 else 0.0
        gW1 = xb.T @ dz1
        gb1 = np.zeros(h1, np.float32)
        for i in range(k):
            for j in range(h1):
                gb1[j] += dz1[i, j]
        for i in range(W2.shape[0]):
            for j in range(out):
                W2[i, j] -= lr * gW2[i, j]
        for j in range(out):
            b2[j] -= lr * gb2[j]
        for i in range(W1.shape[0]):
            for j in range(h1):
                W1[i, j] -= lr * gW1[i, j]
        for j in range(h1):
            b1[j] -= lr * gb1[j]
    return step


def make_model(seed, d_in=784, h1=64, d_out=10):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.05, (d_in, h1)).astype(np.float32)
    b1 = np.zeros(h1, np.float32)
    W2 = rng.normal(0, 0.05, (h1, d_out)).astype(np.float32)
    b2 = np.zeros(d_out, np.float32)
    return W1, b1, W2, b2


def one_cycle_lrs(total_steps, max_lr=0.08, div_factor=25.0, final_div=10000.0, pct_start=0.3):
    initial_lr = max_lr / div_factor
    final_lr = initial_lr / final_div
    up = int(total_steps * pct_start)
    down = max(1, total_steps - up)
    lrs = []
    for i in range(up):
        t = i / max(1, up - 1)
        lrs.append(initial_lr + (max_lr - initial_lr) * t)
    for i in range(down):
        t = i / max(1, down - 1)
        lrs.append(final_lr + (max_lr - final_lr) * (1 + math.cos(math.pi * t)) / 2)
    return np.array(lrs[:total_steps], dtype=np.float32)


def warmup_jit(h1=64):
    X = np.zeros((32, 784), np.float32)
    y = np.zeros(32, np.int64)
    idx = np.arange(32).astype(np.int64)
    W1, b1, W2, b2 = make_model(0, h1=h1)
    lr_arr = np.array([0.03, 0.03], dtype=np.float32)
    train_epoch_nb_lrarr(X, y, W1, b1, W2, b2, lr_arr, 16, idx, 0)
    predict_nb(X, W1, b1, W2, b2)


def accuracy(pred, y):
    return float((pred == y).mean())


def run_one(seed, Xtr, Xte, ytr, yte, q_mins, epochs, batch, max_lr, h1):
    W1, b1, W2, b2 = make_model(seed, h1=h1)
    n = len(Xtr)
    steps_per_epoch = math.ceil(n / batch)
    total_steps = epochs * steps_per_epoch
    oc_lrs = one_cycle_lrs(total_steps, max_lr=max_lr)
    r0 = read_rapl_uj()
    e_at = {q: None for q in q_mins}
    step = 0
    for ep in range(1, epochs + 1):
        idx = np.random.default_rng(seed * 1000 + ep).permutation(n).astype(np.int64)
        step = train_epoch_nb_lrarr(Xtr, ytr, W1, b1, W2, b2, oc_lrs, batch, idx, step)
        pred = predict_nb(Xte, W1, b1, W2, b2)
        te = accuracy(pred, yte)
        r1c = read_rapl_uj()
        for q in q_mins:
            if e_at[q] is None and te >= q:
                e_at[q] = {"epoch": ep, "test_acc": te, "J": (r1c - r0) / 1e6 if r0 is not None and r1c is not None else None}
    r1 = read_rapl_uj()
    te = accuracy(predict_nb(Xte, W1, b1, W2, b2), yte)
    return {
        "seed": seed,
        "test_acc": te,
        "J_total": (r1 - r0) / 1e6 if r0 is not None and r1 is not None else None,
        "e_at": e_at,
    }


def summarize(runs, q_mins):
    out = {"final_acc_median": float(np.median([r["test_acc"] for r in runs]))}
    for q in q_mins:
        passed = [r for r in runs if r["e_at"][q] and r["e_at"][q]["J"] is not None]
        if not passed:
            out[f"q{q}"] = {"pass_rate": 0.0, "median_J": None, "iqr_J": None, "median_epoch": None}
        else:
            Js = [r["e_at"][q]["J"] for r in passed]
            eps = [r["e_at"][q]["epoch"] for r in passed]
            out[f"q{q}"] = {
                "pass_rate": len(passed) / len(runs),
                "median_J": float(np.median(Js)),
                "iqr_J": [float(np.percentile(Js, 25)), float(np.percentile(Js, 75))],
                "median_epoch": float(np.median(eps)),
            }
    return out


def auto_config(task, q_min):
    if task == "mnist-full" or q_min >= 0.95:
        return {"hidden": 128, "batch": 128, "epochs": 20, "max_lr": 0.08}
    return {"hidden": 64, "batch": 64, "epochs": 40, "max_lr": 0.08}


def main():
    ap = argparse.ArgumentParser(description="Reproducible RAPL-measured champion")
    ap.add_argument("--task", choices=["mnist12k", "mnist-full"], default="mnist12k")
    ap.add_argument("--q-min", type=float, default=None, help="Primary quality threshold (default: 0.94 for mnist12k, 0.95 for mnist-full)")
    ap.add_argument("--q-extra", type=float, nargs="*", default=None, help="Extra thresholds to report")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=42, help="Used for mnist12k")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--max-lr", type=float, default=None)
    ap.add_argument("--n-total", type=int, default=12000, help="Used for mnist12k")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    default_q = 0.95 if args.task == "mnist-full" else 0.94
    q_min = args.q_min if args.q_min is not None else default_q
    cfg = auto_config(args.task, q_min)
    hidden = args.hidden if args.hidden is not None else cfg["hidden"]
    batch = args.batch if args.batch is not None else cfg["batch"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    max_lr = args.max_lr if args.max_lr is not None else cfg["max_lr"]

    if args.q_extra is None:
        q_extra = [0.93, 0.94, 0.95]
    else:
        q_extra = args.q_extra
    q_mins = tuple(sorted(set([q_min] + q_extra)))

    if args.task == "mnist-full":
        print("Loading full MNIST (60k/10k standard split)...", flush=True)
        Xtr, Xte, ytr, yte, task_meta = load_mnist_full()
    else:
        print(f"Loading MNIST subset n_total={args.n_total} split_seed={args.split_seed}...", flush=True)
        Xtr, Xte, ytr, yte, task_meta = load_mnist12k(split_seed=args.split_seed, n_total=args.n_total)

    print("JIT warmup (excluded from measurement)...", flush=True)
    t0 = time.time()
    warmup_jit(h1=hidden)
    warmup_s = time.time() - t0
    print(f"JIT warmup done in {warmup_s:.2f}s", flush=True)

    runs = []
    for r in range(args.repeats):
        print(f"Run {r+1}/{args.repeats}...", flush=True)
        res = run_one(r, Xtr, Xte, ytr, yte, q_mins, epochs, batch, max_lr, hidden)
        runs.append(res)
        brief = {str(q): (res["e_at"][q]["J"] if res["e_at"][q] else None) for q in q_mins}
        print(f"  J_to_Q {brief} final_acc={res['test_acc']:.4f}", flush=True)

    report = {
        "config": {
            **task_meta,
            "model": f"784-{hidden}-10",
            "dtype": "float32",
            "optimizer": "backprop_sgd_onecycle",
            "max_lr": max_lr,
            "batch": batch,
            "epochs": epochs,
            "repeats": args.repeats,
            "q_mins": list(q_mins),
            "primary_q_min": q_min,
            "rapl_path": RAPL_PATH,
            "jit_warmup_excluded": True,
            "jit_warmup_s": warmup_s,
        },
        "summary": summarize(runs, q_mins),
        "runs": runs,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
