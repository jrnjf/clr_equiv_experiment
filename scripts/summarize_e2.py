#!/usr/bin/env python3
"""
Summarise an E2 run directory: the table we need to interpret the weight-decay
ablation.

    python scripts/summarize_e2.py results_e2 --config configs/cifar.json

Columns
    wd          literal weight_decay from the config (same for every arm)
    mean shr    mean per-step decay shrink, which differs per convention
    peak acc    best validation accuracy over the run, with its epoch
    first 90    first epoch reaching 90%
    final       accuracy at the last epoch
    drop        peak - final  (positive = degraded after its best epoch)
    loss        final validation loss
    wnorm       mean weight norm at epoch 50 and at the final epoch

`peak acc` and `final` disagree when a run collapses late, which is the whole
point of this ablation -- report both.
"""
import argparse, csv, glob, json, os, statistics as st, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODES = {
    "B1_external_wd_eff": "triangle*cosine",
    "B2_external_wd_base": "lr_max*cosine",
    "F_external_wd_none": "1.0",
    "A_internal": "triangle*cosine",
    "C_cosine_only": "l cosine (no triangle)",
}


def mean_shrink(wd, cfg, mode):
    """Mean per-step decay multiplier for this convention."""
    from schedules import mean_effective_lr
    spe = int(cfg["batch_size"]) and 50000 // int(cfg["batch_size"])  # nominal
    T = cfg["epochs"]
    lr_max, swr = cfg["lr_max"], cfg["lr_SWR"]
    g, end = cfg.get("global_schedule", "cosine"), cfg.get("end_ratio", 1e-4)
    if mode == "triangle*cosine":
        return wd * mean_effective_lr(spe, lr_max, swr, T, g, end)
    if mode == "lr_max*cosine":
        return wd * mean_effective_lr(spe, lr_max, 1.0, T, g, end)
    return wd * 1.0


def load_runs(d):
    runs = []
    for p in glob.glob(os.path.join(d, "*.json")):
        if os.path.basename(p).startswith("e1_summary"):
            try:
                runs.extend(json.load(open(p)))
                continue
            except Exception:
                pass
        try:
            x = json.load(open(p))
            if isinstance(x, dict) and "arm" in x and "val_history" in x:
                runs.append(x)
        except Exception:
            pass
    # dedupe by (arm, seed), preferring the summary file's copy
    uniq = {}
    for r in runs:
        uniq[(r["arm"], r["seed"])] = r
    return list(uniq.values())


def weight_norms(csv_path):
    """Per-epoch mean weight norm, keyed 1-based to match val_history."""
    if not os.path.exists(csv_path):
        return {}
    acc = {}
    for row in csv.DictReader(open(csv_path)):
        try:
            acc.setdefault(int(row["epoch"]) + 1, []).append(float(row["weight_norm"]))
        except (KeyError, ValueError):
            continue
    return {e: st.mean(v) for e, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="results_e2")
    ap.add_argument("--config", default="configs/cifar.json")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    wd = float(cfg["weight_decay"])
    runs = load_runs(a.dir)
    if not runs:
        print(f"no runs with val_history found in {a.dir}/")
        print("expected e1_summary_*.json or <arm>_seed<N>.json there")
        return 1

    print(f"dir={a.dir}  weight_decay={wd:g}  epochs={cfg['epochs']}")
    print(f"{'arm':<24}{'mean shr':>10}{'peak':>8}{'@ep':>5}{'1st90':>7}"
          f"{'final':>8}{'drop':>7}{'loss':>8}{'wn@50':>8}{'wn@end':>8}")
    print("-" * 104)

    by_arm = {}
    for r in sorted(runs, key=lambda x: (x["arm"], x["seed"])):
        hist = r["val_history"]
        accs = [h["val_acc"] for h in hist]
        peak = max(accs)
        pep = accs.index(peak) + 1
        final = accs[-1]
        first90 = next((h["epoch"] for h in hist if h["val_acc"] >= 0.90), None)
        wn = weight_norms(os.path.join(a.dir, f"{r['arm']}_seed{r['seed']}.csv"))
        w50 = wn.get(50)
        wend = wn.get(len(hist))
        mode = MODES.get(r["arm"], "?")
        try:
            shr = mean_shrink(wd, cfg, mode)
        except Exception:
            shr = float("nan")
        by_arm.setdefault(r["arm"], []).append(
            (peak, final, first90, w50, wend, r["final_val_loss"], shr))
        print(f"{r['arm']+f' s{r['seed']}':<24}{shr:>10.2e}{peak:>8.4f}{pep:>5}"
              f"{str(first90):>7}{final:>8.4f}{100*(peak-final):>+7.2f}"
              f"{r['final_val_loss']:>8.4f}"
              f"{(f'{w50:.1f}' if w50 else '-'):>8}"
              f"{(f'{wend:.1f}' if wend else '-'):>8}")

    if any(len(v) > 1 for v in by_arm.values()):
        print("\nper-arm mean over seeds:")
        print(f"{'arm':<24}{'n':>3}{'peak':>8}{'final':>8}{'drop':>7}{'1st90':>7}")
        for arm, rows in by_arm.items():
            print(f"{arm:<24}{len(rows):>3}{st.mean(r[0] for r in rows):>8.4f}"
                  f"{st.mean(r[1] for r in rows):>8.4f}"
                  f"{st.mean(100*(r[0]-r[1]) for r in rows):>+7.2f}"
                  f"{st.mean(r[2] for r in rows if r[2]):>7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
