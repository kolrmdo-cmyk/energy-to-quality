#!/usr/bin/env python3
"""v37: cleaner proof of energy savings on full MNIST.

Same task/split/Q/RAPL as v34/v36, interleaved baseline vs one-cycle,
more repeats, idle-power sampling, wraparound-safe RAPL, per-epoch energy.
5/5-style counts here mean different training seeds on the SAME fixed
standard MNIST split (60k train / 10k test).
"""
from __future__ import annotations
import argparse, json, math, os, time, platform
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from numba import njit

RAPL_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_MAX_PATH = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"

def read_rapl_uj(path=RAPL_PATH):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None

def rapl_max_uj(path=RAPL_MAX_PATH):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None

def rapl_delta_j(r0, r1, max_uj=None):
    if r0 is None or r1 is None:
        return None
    d = r1 - r0
    if d < 0:
        if max_uj is None:
            max_uj = rapl_max_uj()
        if max_uj:
            d += max_uj
        else:
            return None
    return d / 1e6

def sample_idle_power(duration_s=10.0, interval_s=0.5):
    max_uj = rapl_max_uj()
    samples = []
    r_prev = read_rapl_uj()
    t_prev = time.time()
    end = t_prev + duration_s
    while time.time() < end:
        time.sleep(interval_s)
        r = read_rapl_uj()
        t = time.time()
        dj = rapl_delta_j(r_prev, r, max_uj)
        dt = t - t_prev
        if dj is not None and dt > 0:
            samples.append({"dt_s": dt, "dJ": dj, "W": dj / dt})
        r_prev, t_prev = r, t
    if not samples:
        return {"duration_s": duration_s, "samples": 0, "median_W": None, "mean_W": None}
    ws = [s["W"] for s in samples]
    return {
        "duration_s": duration_s,
        "samples": len(samples),
        "median_W": float(np.median(ws)),
        "mean_W": float(np.mean(ws)),
        "iqr_W": [float(np.percentile(ws, 25)), float(np.percentile(ws, 75))],
    }

def system_snapshot():
    snap = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "loadavg": os.getloadavg() if hasattr(os, "getloadavg") else None,
    }
    for p in ["/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
              "/sys/devices/system/cpu/intel_pstate/no_turbo"]:
        try:
            with open(p) as f:
                snap[os.path.basename(p)] = f.read().strip()
        except Exception:
            pass
    snap["rapl_max_energy_range_uj"] = rapl_max_uj()
    return snap

def load_full_mnist():
    d = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = d.data.astype(np.float32); y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = X[:60000], X[60000:], y[:60000], y[60000:]
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    return Xtr, Xte, ytr, yte

@njit(cache=False)
def softmax_nb(z):
    out = np.empty_like(z)
    for i in range(z.shape[0]):
        m = z[i, 0]
        for j in range(1, z.shape[1]):
            if z[i, j] > m: m = z[i, j]
        s = 0.0
        for j in range(z.shape[1]):
            out[i, j] = math.exp(z[i, j] - m); s += out[i, j]
        invs = 1.0 / s
        for j in range(z.shape[1]): out[i, j] *= invs
    return out

@njit(cache=False)
def predict_nb(X, W1, b1, W2, b2):
    h = X @ W1
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h[i, j] + b1[j]; h[i, j] = v if v > 0.0 else 0.0
    z = h @ W2
    for i in range(z.shape[0]):
        for j in range(z.shape[1]): z[i, j] += b2[j]
    p = softmax_nb(z)
    pred = np.empty(X.shape[0], np.int64)
    for i in range(p.shape[0]):
        bi = 0; bv = p[i, 0]
        for j in range(1, p.shape[1]):
            if p[i, j] > bv: bv = p[i, j]; bi = j
        pred[i] = bi
    return pred

@njit(cache=False)
def train_epoch_nb_lrarr(X, y, W1, b1, W2, b2, lr_arr, batch, idx, step_offset):
    n = X.shape[0]; step = step_offset; h1 = W1.shape[1]; out = W2.shape[1]
    for start in range(0, n, batch):
        end = start + batch
        if end > n: end = n
        k = end - start
        lr = lr_arr[step] if step < lr_arr.shape[0] else lr_arr[lr_arr.shape[0] - 1]
        step += 1
        xb = np.empty((k, X.shape[1]), np.float32); yb = np.empty(k, np.int64)
        for ii in range(k):
            src = idx[start + ii]; yb[ii] = y[src]
            for j in range(X.shape[1]): xb[ii, j] = X[src, j]
        z1 = xb @ W1; h = np.empty((k, h1), np.float32)
        for i in range(k):
            for j in range(h1):
                v = z1[i, j] + b1[j]; h[i, j] = v if v > 0.0 else 0.0
        z2 = h @ W2
        for i in range(k):
            for j in range(out): z2[i, j] += b2[j]
        p = softmax_nb(z2); dz2 = p.copy(); invk = np.float32(1.0 / k)
        for i in range(k):
            dz2[i, yb[i]] -= 1.0
            for j in range(out): dz2[i, j] *= invk
        gW2 = h.T @ dz2; gb2 = np.zeros(out, np.float32)
        for i in range(k):
            for j in range(out): gb2[j] += dz2[i, j]
        dh = dz2 @ W2.T; dz1 = np.empty((k, h1), np.float32)
        for i in range(k):
            for j in range(h1): dz1[i, j] = dh[i, j] if h[i, j] > 0.0 else 0.0
        gW1 = xb.T @ dz1; gb1 = np.zeros(h1, np.float32)
        for i in range(k):
            for j in range(h1): gb1[j] += dz1[i, j]
        for i in range(W2.shape[0]):
            for j in range(out): W2[i, j] -= lr * gW2[i, j]
        for j in range(out): b2[j] -= lr * gb2[j]
        for i in range(W1.shape[0]):
            for j in range(h1): W1[i, j] -= lr * gW1[i, j]
        for j in range(h1): b1[j] -= lr * gb1[j]
    return step

def make_nb(seed=0, h1=128):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.05, (784, h1)).astype(np.float32); b1 = np.zeros(h1, np.float32)
    W2 = rng.normal(0, 0.05, (h1, 10)).astype(np.float32); b2 = np.zeros(10, np.float32)
    return W1, b1, W2, b2

def one_cycle_lrs(total_steps, max_lr=0.08, div_factor=25, final_div=10000, pct_start=0.3):
    initial_lr = max_lr / div_factor; final_lr = initial_lr / final_div
    up = int(total_steps * pct_start); down = max(1, total_steps - up); lrs = []
    for i in range(up):
        t = i / max(1, up - 1); lrs.append(initial_lr + (max_lr - initial_lr) * t)
    for i in range(down):
        t = i / max(1, down - 1); lrs.append(final_lr + (max_lr - final_lr) * (1 + math.cos(math.pi * t)) / 2)
    return np.array(lrs[:total_steps], dtype=np.float32)

def warmup(h1=128):
    X = np.zeros((32, 784), np.float32); y = np.zeros(32, np.int64); idx = np.arange(32).astype(np.int64)
    W1, b1, W2, b2 = make_nb(0, h1); lr_arr = np.array([0.03, 0.03], dtype=np.float32)
    train_epoch_nb_lrarr(X, y, W1, b1, W2, b2, lr_arr, 16, idx, 0); predict_nb(X, W1, b1, W2, b2)

def accuracy_np(pred, y): return float((pred == y).mean())

def measure(mode, seed, Xtr, Xte, ytr, yte, q_mins, epochs=20, batch=128, h1=128, max_lr=0.08):
    W1, b1, W2, b2 = make_nb(seed, h1)
    n = len(Xtr); steps_per_epoch = math.ceil(n / batch); total_steps = epochs * steps_per_epoch
    if mode == 'onecycle':
        lr_arr = one_cycle_lrs(total_steps, max_lr=max_lr)
    elif mode == 'const_lr':
        lr_arr = np.full(total_steps, 0.03, dtype=np.float32)
    else:
        raise ValueError(mode)
    max_uj = rapl_max_uj()
    r0 = read_rapl_uj(); t0 = time.time()
    e_at = {q: None for q in q_mins}
    per_epoch = []
    step = 0
    for ep in range(1, epochs + 1):
        idx = np.random.default_rng(seed * 1000 + ep).permutation(n).astype(np.int64)
        step = train_epoch_nb_lrarr(Xtr, ytr, W1, b1, W2, b2, lr_arr, batch, idx, step)
        pred = predict_nb(Xte, W1, b1, W2, b2)
        te = accuracy_np(pred, yte)
        r1c = read_rapl_uj(); t1c = time.time()
        j_cum = rapl_delta_j(r0, r1c, max_uj)
        per_epoch.append({"epoch": ep, "test_acc": te, "J_cum": j_cum, "wall_s_cum": t1c - t0})
        for q in q_mins:
            if e_at[q] is None and te >= q:
                e_at[q] = {"epoch": ep, "test_acc": te, "J": j_cum, "wall_s": t1c - t0}
    r1 = read_rapl_uj(); t1 = time.time()
    te = accuracy_np(predict_nb(Xte, W1, b1, W2, b2), yte)
    return {
        "mode": mode, "seed": seed, "test_acc": te,
        "J_total": rapl_delta_j(r0, r1, max_uj),
        "wall_s_total": t1 - t0,
        "e_at": e_at,
        "per_epoch": per_epoch,
    }

def summarize(runs, q_mins):
    out = {"n": len(runs), "final_acc_median": float(np.median([r['test_acc'] for r in runs]))}
    for q in q_mins:
        passed = [r for r in runs if r['e_at'][q] and r['e_at'][q]['J'] is not None]
        if not passed:
            out[f"q{q}"] = {"pass_rate": 0.0, "n_pass": 0, "median_J": None, "iqr_J": None, "median_epoch": None}
        else:
            Js = [r['e_at'][q]['J'] for r in passed]
            eps = [r['e_at'][q]['epoch'] for r in passed]
            out[f"q{q}"] = {
                "pass_rate": len(passed) / len(runs),
                "n_pass": len(passed),
                "median_J": float(np.median(Js)),
                "iqr_J": [float(np.percentile(Js, 25)), float(np.percentile(Js, 75))],
                "mean_J": float(np.mean(Js)),
                "min_J": float(np.min(Js)),
                "max_J": float(np.max(Js)),
                "median_epoch": float(np.median(eps)),
            }
    return out

def paired_summary(runs_a, runs_b, q, label_a, label_b):
    # pair by seed
    by_seed_a = {r['seed']: r for r in runs_a}
    by_seed_b = {r['seed']: r for r in runs_b}
    seeds = sorted(set(by_seed_a) & set(by_seed_b))
    pairs = []
    for s in seeds:
        ra, rb = by_seed_a[s], by_seed_b[s]
        ja = ra['e_at'][q]['J'] if ra['e_at'][q] else None
        jb = rb['e_at'][q]['J'] if rb['e_at'][q] else None
        if ja is not None and jb is not None:
            pairs.append({"seed": s, label_a: ja, label_b: jb, "diff_a_minus_b": ja - jb, "ratio_a_over_b": ja / jb if jb else None})
    if not pairs:
        return {"q": q, "n_pairs": 0}
    diffs = [p["diff_a_minus_b"] for p in pairs]
    ratios = [p["ratio_a_over_b"] for p in pairs if p["ratio_a_over_b"] is not None]
    return {
        "q": q, "n_pairs": len(pairs),
        "median_diff_a_minus_b_J": float(np.median(diffs)),
        "mean_diff_a_minus_b_J": float(np.mean(diffs)),
        "a_better_count": sum(1 for d in diffs if d < 0),
        "b_better_count": sum(1 for d in diffs if d > 0),
        "equal_count": sum(1 for d in diffs if d == 0),
        "median_ratio_a_over_b": float(np.median(ratios)) if ratios else None,
        "pairs": pairs,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--max-lr", type=float, default=0.08)
    ap.add_argument("--q-min", type=float, default=0.95)
    ap.add_argument("--idle-s", type=float, default=10.0)
    ap.add_argument("--out", default="results/proof.json")
    args = ap.parse_args()

    q_mins = tuple(sorted(set([0.93, 0.94, 0.95, args.q_min])))
    modes = ['const_lr', 'onecycle']

    print("system snapshot...", flush=True)
    snap_before = system_snapshot()
    print("idle power sample...", flush=True)
    idle_before = sample_idle_power(duration_s=args.idle_s)

    print("load full mnist...", flush=True)
    Xtr, Xte, ytr, yte = load_full_mnist()
    print(f"warmup JIT (hidden={args.hidden})...", flush=True)
    warmup(args.hidden)

    report = {
        "config": {
            "task": "mnist_full_60k_10k_standard_split",
            "split_note": "Standard MNIST split: train 60k, test 10k. Fixed for all runs.",
            "repeat_note": "Each repeat uses a different training seed on the SAME fixed split; not different data splits.",
            "model": f"784-{args.hidden}-10",
            "dtype": "float32",
            "optimizer": "backprop_sgd",
            "modes": {
                "const_lr": {"lr": 0.03, "schedule": "constant"},
                "onecycle": {"max_lr": args.max_lr, "schedule": "one_cycle", "div_factor": 25, "final_div": 10000, "pct_start": 0.3},
            },
            "batch": args.batch, "epochs": args.epochs, "repeats": args.repeats,
            "q_mins": list(q_mins), "primary_q_min": args.q_min,
            "rapl_path": RAPL_PATH, "rapl_wraparound_handled": True,
            "interleaved": True,
        },
        "system_before": snap_before,
        "idle_power_before": idle_before,
        "runs": {m: [] for m in modes},
    }

    for r in range(args.repeats):
        for mode in modes:  # interleaved to reduce drift bias
            print(f"[{r+1}/{args.repeats}] {mode}...", flush=True)
            res = measure(mode, r, Xtr, Xte, ytr, yte, q_mins, epochs=args.epochs, batch=args.batch, h1=args.hidden, max_lr=args.max_lr)
            # drop per_epoch in stdout for brevity, keep in report
            report['runs'][mode].append(res)
            brief = {str(q): (res['e_at'][q]['J'] if res['e_at'][q] else None) for q in q_mins}
            print(f"  q->J {brief} acc={res['test_acc']:.4f} J_total={res['J_total']:.2f}", flush=True)
        # checkpoint after each pair
        report['summary'] = {m: summarize(report['runs'][m], q_mins) for m in modes}
        report['paired'] = {str(q): paired_summary(report['runs']['const_lr'], report['runs']['onecycle'], q, 'const_lr', 'onecycle') for q in q_mins}
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print("idle power sample after...", flush=True)
    report['idle_power_after'] = sample_idle_power(duration_s=args.idle_s)
    report['system_after'] = system_snapshot()
    report['summary'] = {m: summarize(report['runs'][m], q_mins) for m in modes}
    report['paired'] = {str(q): paired_summary(report['runs']['const_lr'], report['runs']['onecycle'], q, 'const_lr', 'onecycle') for q in q_mins}

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"summary": report['summary'], "paired_primary": report['paired'].get(str(args.q_min))}, ensure_ascii=False, indent=2))
    print(f"Saved: {args.out}", flush=True)

if __name__ == '__main__':
    main()
