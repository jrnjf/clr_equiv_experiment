#!/usr/bin/env python3
"""
E1 -- Implementation-equivalence experiment for DualScheduledSGD.

Question
--------
Does DualScheduledSGD's internal per-batch triangular schedule reproduce the
behaviour of the same schedule applied externally to a stock SGD optimizer?

Arms
----
A    internal   DualScheduledSGD, triangular x cosine, wd on effective LR
B1   external   torch.optim.SGD + ExternalDualScheduler, wd on effective LR
                 -> must match A exactly (same math, independent implementation)
B2   external   as B1 but weight decay against the BASE lr only
                 -> isolates the wd-coupling difference
C    cosine     cosine anneal only, no triangular component (lr_SWR=1)
                 -> isolates the value of the cyclic component

Backends
--------
--backend tiny    CPU, synthetic data, seconds. Correctness only, no science.
--backend cifar   CIFAR-10 + CIFAR ResNet-18 (GPU). The real run.

Modes
-----
--dry-run            validate config + LR-series equivalence (torch-free)
--check-equivalence  train A and B1 from the same seed and compare final
                     PARAMETER TRAJECTORIES -- the strongest equivalence claim.
                     Run this before spending GPU time.

Decision rule
-------------
A vs B1 is pass/fail. If parameters agree, the implementation is correct and you
may interpret the rest. If not, there is a bug; do not interpret B2/C.
Then:
  B1 vs C   -> does the cyclic component add anything over annealing alone?
  B1 vs B2  -> does decay-on-effective-LR matter?

Usage
-----
    python experiments/run_equivalence.py --config configs/tiny.json --dry-run
    python experiments/run_equivalence.py --config configs/tiny.json --check-equivalence
    python experiments/run_equivalence.py --config configs/tiny.json
    python experiments/run_equivalence.py --config configs/cifar.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dual_scheduler import (assert_same_lr_series, build_constant,
                            build_cosine_only, build_internal, build_method,
                            lr_series, train_step)
from schedules import effective_lr_series

ARMS = ("A_internal", "B1_external_wd_eff", "C_cosine_only", "constant")

DEFAULTS = {
    "backend": "tiny",
    "epochs": 3,
    "lr_max": 0.8,
    "lr_SWR": 100.0,
    "weight_decay": 5e-4,
    "global_schedule": "cosine",
    "end_ratio": 0.01,
    "seeds": [0],
    "device": "cpu",
    "num_workers": 0,
    "batch_size": 128,
    "out_dir": "results",
    "log_every": 1,
}


def load_config(path):
    cfg = dict(DEFAULTS)
    if not path:
        return cfg
    with open(path) as fh:
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError:
                raise SystemExit("PyYAML missing; use a .json config")
            cfg.update(yaml.safe_load(fh) or {})
        else:
            cfg.update(json.load(fh))
    return cfg


def make_opt_builder(cfg, spe, arm):
    # Every arm gets the SAME literal weight_decay from the config. The arms
    # differ only in what that coefficient is multiplied by:
    #     B1 effective : wd * triangle(step) * cosine(epoch)
    #     B2 base      : wd * lr_max * cosine(epoch)
    #     F  none      : wd
    # They therefore have different MEAN decay strengths. That is accepted
    # deliberately: it is the literal "what happens if you set weight_decay=X"
    # comparison. (An automatic strength-matching layer was tried and removed as
    # over-engineering; see git history if it is ever wanted again.)
    common = dict(steps_per_epoch=spe, lr_max=cfg["lr_max"],
                  lr_SWR=cfg["lr_SWR"], global_name=cfg["global_schedule"],
                  total_epochs=cfg["epochs"], end_ratio=cfg["end_ratio"],
                  weight_decay=float(cfg["weight_decay"]),
                  momentum=cfg.get("momentum", 0.0))

    def build(model):
        if arm == "A_internal":
            return build_internal(model, wd_on_effective_lr=True, **common)
        if arm == "B1_external_wd_eff":
            return build_method(model, **common)

        if arm == "C_cosine_only":
            return build_cosine_only(model, **common)
        if arm == "constant":
            return build_constant(model, **common)
        raise ValueError(arm)

    return build


# ---------------------------------------------------------------------------
def dry_run(cfg, allow_odd=False):
    # len(DataLoader(train, batch_size=bs, drop_last=True)) == n_train // bs.
    # Computed here rather than by building the loader so --dry-run stays
    # torch-free; main() always uses the real len(train_loader).
    n_train = int(cfg.get("tiny_train", 2048)) if cfg["backend"] == "tiny" else 50000
    spe = n_train // int(cfg["batch_size"])
    print(f"backend={cfg['backend']} batch={cfg['batch_size']} "
          f"steps_per_epoch={spe} epochs={cfg['epochs']}")
    print(f"lr_max={cfg['lr_max']} lr_SWR={cfg['lr_SWR']} "
          f"global={cfg['global_schedule']} end_ratio={cfg['end_ratio']}")
    if spe % 2:
        msg = (f"steps_per_epoch={spe} is ODD, so the triangular half-cycle is "
               f"{spe/2} and no CyclicLR configuration reproduces the internal "
               f"wave exactly (measured max|diff| ~1.5e-2).")
        if not allow_odd:
            raise SystemExit(msg + " Choose an even step count, or pass "
                                   "--allow-odd when replicating an existing "
                                   "protocol whose step count you cannot change.")
        print(f"WARNING: {msg} Proceeding with --allow-odd; the realized peak "
              f"will be ~{(1 - 1/(spe+1))*100:.1f}% below nominal.")

    worst = assert_same_lr_series(spe, cfg["lr_max"], cfg["lr_SWR"],
                                  n_steps=min(spe * cfg["epochs"], 3000),
                                  global_name=cfg["global_schedule"],
                                  total_epochs=cfg["epochs"],
                                  end_ratio=cfg["end_ratio"])
    print(f"\nLR series internal vs external: max|diff| = {worst:.3e} -> PASS")

    series = lr_series(spe, cfg["lr_max"], cfg["lr_SWR"], spe * cfg["epochs"],
                       cfg["epochs"], cfg["end_ratio"],
                       cfg["global_schedule"])
    ep0 = series[:spe]
    print(f"  first={series[0]:.6f} peak_ep0={max(ep0):.5f} "
          f"last={series[-1]:.6f} mean={sum(series)/len(series):.5f}")
    print("\nconfig valid. Run without --dry-run to train.")
    return 0


def check_equivalence(cfg):
    """Train A and B1 from identical init/data order; compare final parameters."""
    import torch
    from backends import build as build_backend

    if cfg["backend"] != "tiny":
        cfg = dict(cfg)
        cfg["backend"] = "tiny"
        print("NOTE: --check-equivalence forces backend=tiny (fast, CPU)")

    train_loader, _val, _test, make_model, eval_fn, spe = build_backend(cfg)
    print(f"tiny backend: steps_per_epoch={spe} epochs={cfg['epochs']}")

    finals = {}
    for arm in ("A_internal", "B1_external_wd_eff"):
        build_opt = make_opt_builder(cfg, spe, arm)
        model = make_model().to(cfg["device"])
        opt, sched = build_opt(model)
        crit = torch.nn.CrossEntropyLoss()
        for _ in range(cfg["epochs"]):
            model.train()
            for step_in_epoch, (x, y) in enumerate(train_loader):
                if step_in_epoch >= spe:
                    break
                x, y = x.to(cfg["device"]), y.to(cfg["device"])
                opt.zero_grad(set_to_none=True)
                loss = crit(model(x), y)
                loss.backward()
                train_step(opt, sched, loss)
        finals[arm] = [p.detach().clone() for p in model.parameters()]
        print(f"  trained {arm}")

    diffs = [float((a - b).abs().max())
             for a, b in zip(finals["A_internal"], finals["B1_external_wd_eff"])]
    worst = max(diffs)
    scale = max(float(p.abs().max()) for p in finals["A_internal"])
    relative = worst / max(scale, 1e-12)

    print(f"\nfinal-parameter max|A - B1| = {worst:.3e} "
          f"(scale {scale:.3e}, relative {relative:.2e})")

    # Tolerance is RELATIVE, not absolute. Rationale, measured on the tiny
    # backend (12 epochs, 16 steps/epoch):
    #   float32 -> 1.4e-6 absolute (6.3e-6 relative)
    #   float64 -> 6.1e-16 (machine zero)
    # so the two implementations are algebraically identical and the residual
    # is pure float32 accumulation. An absolute 1e-6 threshold falsely failed
    # at 1.416e-6; a genuine algorithmic bug shows up at ~1e-2 relative
    # (the ordering bug measured 7.6e-2, the decay-ordering bug 4e-3), so a
    # 1e-4 relative gate separates float noise from real disagreement by >1
    # order of magnitude on each side.
    TOL_REL = 1e-4
    if relative <= TOL_REL:
        print(f"EQUIVALENT within float32 tolerance (rel {relative:.2e} <= {TOL_REL:.0e}).")
        print("  (float64 reruns give ~1e-16: the difference is rounding, not logic.)")
        return 0
    print(f"NOT EQUIVALENT: relative {relative:.2e} > {TOL_REL:.0e}. "
          f"Investigate before interpreting any other arm.")
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-equivalence", action="store_true")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--out-dir", default=None,
                    help="override cfg['out_dir'] (keeps runs separated)")
    ap.add_argument("--lr-max", type=float, default=None,
                    help="override cfg['lr_max'] (for a constant-LR sweep)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="override cfg['seeds'] (e.g. --seeds 0 for a pilot)")
    ap.add_argument("--allow-odd", action="store_true",
                    help="permit an odd steps_per_epoch (peak is then slightly "
                         "undersampled); use when replicating a fixed protocol")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    if args.lr_max is not None:
        cfg["lr_max"] = args.lr_max

    if args.dry_run:
        return dry_run(cfg, allow_odd=args.allow_odd)
    if args.check_equivalence:
        return check_equivalence(cfg)

    import torch
    from backends import build as build_backend
    from trainer import run_arm

    if cfg.get("device") == "cuda" and not torch.cuda.is_available():
        print("WARNING: cuda unavailable, falling back to cpu")
        cfg["device"] = "cpu"

    train_loader, _val, _test, make_model, eval_fn, spe = build_backend(cfg)
    print(f"{cfg['backend']}: steps_per_epoch={spe} epochs={cfg['epochs']} "
          f"arms={args.arms} seeds={cfg['seeds']} device={cfg['device']}")

    results = []
    for seed in cfg["seeds"]:
        for arm in args.arms:
            # "constant" sweeps over lr_max, so fold the LR into the run name;
            # otherwise every LR would overwrite the same (arm, seed) entry.
            name = (f"constant_lr{cfg['lr_max']:g}" if arm == "constant" else arm)
            print(f"\n=== {name} | seed {seed} ===")
            results.append(run_arm(
                arm_name=name, make_model=make_model,
                make_optimizer_and_scheduler=make_opt_builder(cfg, spe, arm),
                train_loader=train_loader, eval_fn=eval_fn,
                epochs=cfg["epochs"], steps_per_epoch=spe,
                device=cfg["device"], log_every=cfg["log_every"],
                out_dir=cfg["out_dir"], seed=seed))

    os.makedirs(cfg["out_dir"], exist_ok=True)
    # MERGE with any existing summary rather than overwrite: runs are launched
    # in several batches (different arms, different seeds), and clobbering the
    # file would silently discard earlier results.
    os.makedirs(cfg["out_dir"], exist_ok=True)
    out = os.path.join(cfg["out_dir"], f"e1_summary_{cfg['backend']}.json")
    merged = {}
    if os.path.exists(out):
        try:
            for r in json.load(open(out)):
                merged[(r.get("arm"), r.get("seed"))] = r
        except (json.JSONDecodeError, TypeError):
            print(f"warning: {out} unreadable, starting a fresh summary")
    for r in results:
        merged[(r.get("arm"), r.get("seed"))] = r
    with open(out, "w") as fh:
        json.dump(list(merged.values()), fh, indent=2)
    print(f"\nwrote {out} ({len(merged)} runs total, "
          f"{len(results)} from this invocation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
