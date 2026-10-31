#!/usr/bin/env python3
"""
Decompose per-step time so the real bottleneck can be identified.

The logging fix turned out to be only ~11% of the gap (see docs/PERFORMANCE.md),
so this script measures the pieces separately:

    data loading | forward | backward | optimizer step | epoch eval

Run on the GPU box:
    python scripts/profile_step.py --config configs/cifar.json
    python scripts/profile_step.py --config configs/cifar.json --batches 60

Every segment is wrapped in torch.cuda.synchronize() so the numbers are real
device time, not queue-submission time.
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backends import build as build_backend
from dual_scheduler import build_external


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, n):
    """Run fn once per iteration for n iterations; return ms/iteration."""
    fn()                       # warmup (allocators, cudnn autotune)
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cifar.json")
    ap.add_argument("--batches", type=int, default=40)
    a = ap.parse_args()

    import torch
    cfg = json.load(open(a.config))
    if not torch.cuda.is_available():
        print("CUDA unavailable; timings will not reflect GPU behaviour.")
    train_loader, _v, _t, make_model, eval_fn, spe = build_backend(cfg)
    device = cfg.get("device", "cuda")

    model = make_model().to(device)
    crit = torch.nn.CrossEntropyLoss()
    opt, sched = build_external(
        model, steps_per_epoch=spe, lr_max=cfg["lr_max"],
        lr_SWR=cfg["lr_SWR"], global_name=cfg["global_schedule"],
        total_epochs=cfg["epochs"], end_ratio=cfg["end_ratio"],
        weight_decay=cfg["weight_decay"],
        wd_on_effective_lr=True, momentum=cfg.get("momentum", 0.0))

    batches = []
    for i, (x, y) in enumerate(train_loader):
        batches.append((x, y))
        if len(batches) >= a.batches:
            break
    print(f"torch {torch.__version__} | cuda {torch.version.cuda}")
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({pr.total_memory/1e9:.1f} GB, "
              f"{pr.multi_processor_count} SMs)")
    print(f"device={device} batch={cfg['batch_size']} steps/epoch={spe} "
          f"measuring on {len(batches)} batches\n")

    # ---- 0. DataLoader alone: only next(), no .to(), no compute.
    # This is where waiting hides in the training loop: the `for ... in loader`
    # fetch sits outside every per-operation timer in an instrumented trainer.
    def loader_only():
        n = 0
        for _x, _y in train_loader:
            n += 1
            if n >= len(batches):
                break
    ms_loader = timeit(loader_only, 2) / len(batches)

    # ---- 1. data only: pull tensors to device, no compute
    def data_only():
        for x, y in batches:
            x.to(device, non_blocking=True)
            y.to(device, non_blocking=True)
    ms_data = timeit(data_only, 3) / len(batches)

    # ---- 2. forward only
    x0, y0 = batches[0][0].to(device), batches[0][1].to(device)
    model.eval()
    def fwd():
        with torch.no_grad():
            model(x0)
    ms_fwd = timeit(fwd, 10)

    # ---- 3. forward + backward
    model.train()
    def fwd_bwd():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x0), y0)
        loss.backward()
    ms_fb = timeit(fwd_bwd, 10)

    # ---- 4. + optimizer step (the arm under test)
    def full_step():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x0), y0)
        loss.backward()
        sched.prep()
        opt.step()
        sched.advance()
    ms_full = timeit(full_step, 10)

    # ---- 5. same, but with a stock optimizer only (no scheduler calls)
    plain = torch.optim.SGD(model.parameters(), lr=cfg["lr_max"] / cfg["lr_SWR"],
                            momentum=cfg.get("momentum", 0.0))
    def full_step_plain():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x0), y0)
        loss.backward()
        plain.step()
    ms_plain = timeit(full_step_plain, 10)

    # ---- 6. epoch evaluation
    def do_eval():
        eval_fn(model, None)
    ms_eval = timeit(do_eval, 2)

    print(f"{'segment':<34}{'ms':>10}")
    print(f"{'DataLoader only (next())':<34}{ms_loader:>10.2f}")
    print(f"{'device copy (.to)':<34}{ms_data:>10.2f}")
    print(f"{'forward':<34}{ms_fwd:>10.2f}")
    print(f"{'forward + backward':<34}{ms_fb:>10.2f}")
    print(f"{'  (backward alone)':<34}{ms_fb - ms_fwd:>10.2f}")
    print(f"{'+ stock SGD step':<34}{ms_plain:>10.2f}")
    print(f"{'+ scheduler prep/advance':<34}{ms_full:>10.2f}")
    print(f"{'  (scheduler overhead)':<34}{ms_full - ms_plain:>10.2f}")
    print(f"{'epoch eval (whole split)':<34}{ms_eval:>10.2f}")
    print(f"\nprojected per epoch: "
          f"{(ms_full + ms_data) * spe / 1000 + ms_eval / 1000:.1f} s "
          f"({spe} steps + 1 eval)")
    print(f"  of which eval: {ms_eval / 1000:.1f} s")
    print(f"  of which steps: {(ms_full + ms_data) * spe / 1000:.1f} s")

    # Reconciliation: sum of the measured segments vs a full epoch at that rate.
    measured = (ms_loader + ms_full) * spe / 1000 + ms_eval / 1000
    print(f"\nreconciliation: ({ms_loader:.1f} loader + {ms_full:.1f} step) x {spe} "
          f"+ {ms_eval:.0f} eval = {measured:.1f} s/epoch")
    print(f"  if the real epoch time is much larger, the gap is async submission "
          f"(these numbers are device-synchronised, so it should not be).")

    if ms_loader > 0.5 * ms_full:
        print(f"\n>> DataLoader is the bottleneck: {ms_loader:.1f} ms/batch waited "
              f"for workers. Try --num-workers 0/2/8 and set "
              f"torch.backends.cudnn.benchmark=True.")
    if ms_data > 0.5 * (ms_fb + ms_data):
        print("\n>> data loading looks like a large share; check num_workers / "
              "CPU-side augmentation.")
    if ms_fb - ms_fwd < 0.2 * ms_fb:
        print("\n>> suspicious: backward much cheaper than expected; check that "
              "graphs are not being freed early.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
