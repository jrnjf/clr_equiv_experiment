#!/usr/bin/env python3
"""
Measure the per-step cost of logging, on the real device.

The trainer's per-step metrics force CUDA synchronisation: every float()/.item()
on a GPU tensor blocks until the queue drains. `log_every` now gates the whole
computation, so this script quantifies what that buys on your hardware.

Run on the GPU box:
    python scripts/bench_logging.py --config configs/cifar.json
    python scripts/bench_logging.py --config configs/tiny.json      # CPU sanity

Output: ms/step for several log_every values. A huge log_every effectively logs
nothing, so its number is the "pure training step" cost and the difference is the
logging overhead.

Note: this trains a few real epochs, so on CIFAR it takes ~4 epochs of GPU time
(and downloads CIFAR-10 on first run).
"""
import argparse, json, os, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backends import build as build_backend
from trainer import run_arm

DEFAULTS = dict(lr_max=0.8, lr_SWR=1000.0, weight_decay=5e-4,
                global_schedule="cosine", end_ratio=1e-4, device="cuda",
                num_workers=4, batch_size=128, data_root="./data")


def make_builder(cfg, spe):
    from dual_scheduler import build_external
    return lambda model: build_external(
        model, steps_per_epoch=spe, lr_max=cfg["lr_max"], lr_SWR=cfg["lr_SWR"],
        global_name=cfg["global_schedule"], total_epochs=cfg["epochs"],
        end_ratio=cfg["end_ratio"], weight_decay=cfg["weight_decay"],
        wd_on_effective_lr=True, momentum=cfg.get("momentum", 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cifar.json")
    ap.add_argument("--epochs", type=int, default=2,
                    help="epochs per setting (keep small)")
    ap.add_argument("--log-every", nargs="*", type=int, default=[1, 10, 50, 10**9])
    a = ap.parse_args()

    cfg = dict(DEFAULTS)
    with open(a.config) as fh:
        cfg.update(json.load(fh))
    if cfg.get("device") == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("CUDA unavailable -> running on CPU; sync costs will not show.")
            cfg["device"] = "cpu"
    cfg["epochs"] = a.epochs

    train_loader, _v, _t, make_model, eval_fn, spe = build_backend(cfg)
    print(f"device={cfg['device']} batch={cfg['batch_size']} "
          f"steps/epoch={spe} epochs/setting={a.epochs}\n")

    results = {}
    for le in a.log_every:
        label = "none" if le >= 10**8 else str(le)
        with tempfile.TemporaryDirectory() as td:
            s = run_arm(
                arm_name="bench", make_model=make_model,
                make_optimizer_and_scheduler=make_builder(cfg, spe),
                train_loader=train_loader, eval_fn=eval_fn,
                epochs=a.epochs, steps_per_epoch=spe, device=cfg["device"],
                log_every=le, out_dir=td, seed=0)
        results[label] = s
        print(f"  log_every={label:<6} {s['ms_per_step']:7.1f} ms/step  "
              f"({s['sec_per_epoch']:.1f} s/epoch)")

    base = results.get("none", {}).get("ms_per_step")
    if base:
        print(f"\noverhead relative to no logging (baseline {base:.1f} ms/step):")
        for label, s in results.items():
            r = s["ms_per_step"] / base
            print(f"  log_every={label:<6} {r:5.2f}x  "
                  f"(+{s['ms_per_step'] - base:.1f} ms/step)")
        print("\nProjected 200-epoch run time at each setting:")
        for label, s in results.items():
            print(f"  log_every={label:<6} {s['sec_per_epoch'] * 200 / 3600:6.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
