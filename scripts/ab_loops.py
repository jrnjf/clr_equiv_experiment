#!/usr/bin/env python3
"""
A/B the training loop against the sgd_jumper reference on the SAME machine.

The 200-epoch E1 run measured 94.5 ms/step with logging fully disabled, versus
~45 ms/step implied by the sgd_jumper benchmark CSV. Both are claimed to be the
same machine, so this script isolates the difference:

  A  our loop        stock SGD + ExternalDualScheduler (what E1/E2 use)
  B  their loop      Jumper optimizer, exactly as jumper_expirement.py builds it
  C  bare loop       stock SGD, no scheduler at all  (the compute floor)

All three run on the SAME model instance, the SAME prefetched batches, and the
SAME device, so data loading and model construction are excluded. A second pass
times the real DataLoader to show the data cost separately.

Run on the GPU box:
    python scripts/ab_loops.py --config configs/cifar.json --batches 30
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

REF = os.path.join(os.path.dirname(ROOT), ".sgd_jumper_ref")


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, n):
    fn(); sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cifar.json")
    ap.add_argument("--batches", type=int, default=30)
    a = ap.parse_args()

    import torch
    from backends import build as build_backend
    from dual_scheduler import build_external

    cfg = json.load(open(os.path.join(ROOT, a.config)))
    device = cfg.get("device", "cuda")

    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"cudnn.benchmark={torch.backends.cudnn.benchmark} | "
          f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    else:
        print("CUDA unavailable -> timings will not reflect GPU behaviour")

    train_loader, _v, _t, make_model, eval_fn, spe = build_backend(cfg)
    crit = torch.nn.CrossEntropyLoss()

    # --- model once, shared by all arms
    torch.manual_seed(0)
    model = make_model().to(device)

    # --- prefetch batches so data loading is excluded from loop timing
    batches = []
    for x, y in train_loader:
        batches.append((x.to(device, non_blocking=True),
                        y.to(device, non_blocking=True)))
        if len(batches) >= a.batches:
            break
    print(f"device={device} batch={cfg['batch_size']} steps/epoch={spe} "
          f"batches timed={len(batches)}\n")

    # --- A: our loop
    opt_a, sch_a = build_external(
        model, steps_per_epoch=spe, lr_max=cfg["lr_max"], lr_SWR=cfg["lr_SWR"],
        global_name=cfg["global_schedule"], total_epochs=cfg["epochs"],
        end_ratio=cfg["end_ratio"], weight_decay=cfg["weight_decay"],
        wd_on_effective_lr=True, momentum=cfg.get("momentum", 0.0))

    def loop_a():
        for x, y in batches:
            opt_a.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            sch_a.prep()
            opt_a.step()
            sch_a.advance()
    ms_a = timeit(loop_a, 3) / len(batches)

    # --- C: bare stock SGD, no scheduler
    opt_c = torch.optim.SGD(model.parameters(),
                            lr=cfg["lr_max"] / cfg["lr_SWR"],
                            momentum=cfg.get("momentum", 0.0))

    def loop_c():
        for x, y in batches:
            opt_c.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt_c.step()
    ms_c = timeit(loop_c, 3) / len(batches)

    # --- B: their Jumper, exactly as their benchmark builds it
    ms_b = None
    if os.path.isdir(REF):
        sys.path.insert(0, REF)
        try:
            from sgd_jumper import Jumper
            opt_b = Jumper(model.parameters(), steps_per_epoch=spe, lr=0.8,
                           gamma=20.0, fit_type="log", lr_SWR=10,
                           momentum=0.0, weight_decay=5e-4)

            def loop_b():
                for x, y in batches:
                    opt_b.zero_grad()
                    loss = crit(model(x), y)
                    loss.backward()
                    opt_b.step()
            ms_b = timeit(loop_b, 3) / len(batches)
        except Exception as e:            # noqa: BLE001
            print(f"could not import their Jumper: {e}")
    else:
        print(f"note: {REF} not found; skipping arm B")

    # --- their loop but with stock SGD (isolates optimizer from loop style)
    opt_d = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                            weight_decay=5e-4)

    def loop_d():
        for x, y in batches:
            opt_d.zero_grad()             # their zero_grad style, not set_to_none
            loss = crit(model(x), y)
            loss.backward()
            opt_d.step()
    ms_d = timeit(loop_d, 3) / len(batches)

    print(f"{'arm':<46}{'ms/step':>10}")
    print(f"{'A  our loop (SGD + external scheduler)':<46}{ms_a:>10.2f}")
    if ms_b is not None:
        print(f"{'B  their loop (Jumper)':<46}{ms_b:>10.2f}")
    print(f"{'C  bare loop (stock SGD, no scheduler)':<46}{ms_c:>10.2f}")
    print(f"{'D  their loop style + stock SGD(m=0.9,lr=.1)':<46}{ms_d:>10.2f}")
    print(f"\nA/C ratio (our overhead over compute floor): {ms_a/ms_c:.2f}x")
    if ms_b is not None:
        print(f"A/B ratio (ours vs theirs):                  {ms_a/ms_b:.2f}x")
        print(f"B/C ratio (their optimizer over floor):      {ms_b/ms_c:.2f}x")
    print(f"evaluation per epoch: {timeit(lambda: eval_fn(model, None), 2):.0f} ms")

    # --- data loading alone: iterate the real loader, no compute at all.
    # The A/B above prefetches batches, so it cannot see a loader bottleneck.
    def data_pass():
        n = 0
        for x, y in train_loader:
            x.to(device, non_blocking=True)
            y.to(device, non_blocking=True)
            n += 1
            if n >= len(batches):
                break
    ms_data = timeit(data_pass, 2) / len(batches)
    print(f"\ndata loading alone: {ms_data:.2f} ms/batch "
          f"(num_workers={cfg.get('num_workers')})")
    print(f"  as a share of arm A: {ms_data / ms_a * 100:.0f}%")
    if ms_data > 0.5 * ms_a:
        print("  >> the loader alone costs more than half of a training step; "
              "try --num-workers 0/2/8 to see if it is the bottleneck")
    return 0


if __name__ == "__main__":
    sys.exit(main())
