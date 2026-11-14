"""
Two interchangeable backends.

  tiny   -- CPU, synthetic data, seconds to run. Purpose: prove the code and the
            two scheduling paths are correct. Never intended to produce a
            scientific result.
  cifar  -- CIFAR-10 + CIFAR-adapted ResNet-18 on GPU. The real experiment.

Both expose the same interface:

    build(cfg) -> (train_loader, val_loader, test_loader,
                   make_model, eval_fn, steps_per_epoch)

so experiments/run_equivalence.py is backend-agnostic.
"""

from __future__ import annotations

import math
import os


# ---------------------------------------------------------------------------
# tiny: synthetic, CPU, deterministic
# ---------------------------------------------------------------------------
def _build_tiny(cfg):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    g = torch.Generator().manual_seed(cfg.get("data_seed", 0))

    n_train = int(cfg.get("tiny_train", 2048))
    n_val = int(cfg.get("tiny_val", 512))
    n_test = int(cfg.get("tiny_test", 512))
    dim = int(cfg.get("tiny_dim", 32))
    n_classes = int(cfg.get("tiny_classes", 10))
    bs = int(cfg["batch_size"])

    # ONE fixed set of class centers shared by all splits. Generating fresh
    # centers per split (the first version of this file) makes train and val
    # different tasks, so validation accuracy sits at chance and the smoke test
    # can never distinguish the arms.
    centers = torch.randn(n_classes, dim, generator=g)

    def make_split(n, seed):
        gg = torch.Generator().manual_seed(seed)
        y = torch.randint(0, n_classes, (n,), generator=gg)
        x = centers[y] + 0.9 * torch.randn(n, dim, generator=gg)
        return TensorDataset(x, y)

    train_full = make_split(n_train, 1)
    val_full = make_split(n_val, 2)
    test_full = make_split(n_test, 3)

    # Mirror the cifar backend's val_fraction so the validation/test separation
    # (reviewer 3.4) can be exercised on CPU.
    tiny_val_fraction = float(cfg.get("val_fraction", 0.0))
    if tiny_val_fraction > 0:
        import torch as _t
        from torch.utils.data import Subset
        n = len(train_full)
        n_v = max(1, int(round(n * tiny_val_fraction)))
        gg = _t.Generator().manual_seed(int(cfg.get("split_seed", 0)))
        perm = _t.randperm(n, generator=gg).tolist()
        val_idx, tr_idx = perm[:n_v], perm[n_v:]
        val_full = Subset(train_full, val_idx)
        train_full = Subset(train_full, tr_idx)

    steps = len(train_full) // bs
    # No even-step requirement: the schedule is a LambdaLR and works at any step
    # count (see the note in _build_cifar_like).

    loader_kw = dict(num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_full, batch_size=bs, shuffle=True,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(val_full, batch_size=256, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_full, batch_size=256, shuffle=False, **loader_kw)

    hidden = int(cfg.get("tiny_hidden", 64))

    def make_model():
        torch.manual_seed(cfg.get("init_seed", 0))
        return nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    device = cfg.get("device", "cpu")

    def _eval_on(model, loader):
        model.eval()
        crit = nn.CrossEntropyLoss()
        tot_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                tot_loss += crit(out, y).item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                n += x.size(0)
        return tot_loss / n, correct / n

    def eval_fn(model, _=None):
        return _eval_on(model, val_loader)

    tiny_test_eval = ((lambda model: _eval_on(model, test_loader))
                      if tiny_val_fraction > 0 else None)
    return (train_loader, val_loader, test_loader,
            make_model, eval_fn, len(train_loader), tiny_test_eval)


# ---------------------------------------------------------------------------
# cifar: CIFAR-10 + ResNet-18, GPU
# ---------------------------------------------------------------------------
# Trains on the FULL 50,000-image training set and evaluates on the official
# 10,000-image TEST split. There is deliberately no validation holdout: carving
# one out of train both discards training data and changes steps_per_epoch,
# which in turn changes the triangular period (the schedule depends on it).
def _build_cifar(cfg):
    return _build_cifar_like(cfg, dataset="CIFAR10", num_classes=10)


def _build_cifar100(cfg):
    return _build_cifar_like(cfg, dataset="CIFAR100", num_classes=100)


def _build_cifar_like(cfg, dataset="CIFAR10", num_classes=10):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.models import resnet18

    mean = tuple(cfg.get("mean", (0.4914, 0.4822, 0.4465)))
    std = tuple(cfg.get("std", (0.2470, 0.2435, 0.2616)))

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean, std)])

    root = cfg["data_root"]
    ds = getattr(datasets, dataset)
    train_set = ds(root, train=True, download=True, transform=train_tf)
    test_set = ds(root, train=False, download=True, transform=eval_tf)

    # Reviewer 3.4: hyperparameters must not be selected on the test split.
    # With val_fraction > 0, hold out that fraction of the TRAINING data as a
    # validation set. Two dataset instances are built so the validation split gets
    # eval transforms (no random crop/flip) rather than train transforms.
    test_eval = None
    val_fraction = float(cfg.get("val_fraction", 0.0))
    if val_fraction > 0:
        import torch
        from torch.utils.data import Subset
        n_total = len(train_set)
        n_val = int(round(n_total * val_fraction))
        g = torch.Generator().manual_seed(int(cfg.get("split_seed", 0)))
        perm = torch.randperm(n_total, generator=g).tolist()
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        train_eval_ds = ds(root, train=True, download=True, transform=eval_tf)
        train_set = Subset(train_set, tr_idx)
        val_set = Subset(train_eval_ds, val_idx)

    bs = int(cfg["batch_size"])
    nw = cfg.get("num_workers", 4)
    # persistent_workers=True: without it every `for ... in loader` spawns a
    # fresh worker pool, and that startup is charged to EVERY epoch. Measured
    # on the GPU box: DataLoader-only iteration cost 264 ms/batch over a 40-batch
    # window, which is ~10 s of process startup, and epoch evaluation measured
    # 10.7 s for 20 batches -- i.e. the eval "cost" was almost entirely worker
    # spawn. Together those two accounted for ~20 s of a 36.7 s epoch.
    pw = nw > 0
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=pw)
    # test set doubles as the evaluation set
    eval_loader = DataLoader(test_set, batch_size=1024, shuffle=False,
                             num_workers=nw, pin_memory=True,
                             persistent_workers=pw)

    if val_fraction > 0:
        val_loader = DataLoader(val_set, batch_size=1024, shuffle=False,
                                num_workers=nw, pin_memory=True,
                                persistent_workers=pw)
    else:
        val_loader = eval_loader

    # NOTE: no even-step requirement. The schedule is a plain LambdaLR, which
    # reproduces the analytic triangle x cosine series at any steps_per_epoch
    # (assert_lambdalr_alignment checks this). The old guard existed only because
    # the E1 equivalence check compared against CyclicLR; that path is gone, and the
    # guard blocked the 45k/5k validation-split runs (351 steps, odd).
    steps = len(train_loader)

    def make_model():
        m = resnet18(num_classes=num_classes)
        m.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m

    device = cfg.get("device", "cuda")

    def _eval_on(model, loader):
        model.eval()
        crit = nn.CrossEntropyLoss()
        tot_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                tot_loss += crit(out, y).item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                n += x.size(0)
        return tot_loss / n, correct / n

    def eval_fn(model, _=None):
        return _eval_on(model, val_loader)

    if val_fraction > 0:
        test_eval = lambda model: _eval_on(model, eval_loader)

    return (train_loader, val_loader, eval_loader, make_model, eval_fn,
            len(train_loader), test_eval)


# ---------------------------------------------------------------------------
# cifar_bench: EXACT replica of the sgd_jumper benchmark protocol
# ---------------------------------------------------------------------------
# Differences from `cifar` above, all taken from jrnjf/sgd_jumper utils.py:
#   * batch_size 128 on the FULL 50,000 training images  -> 390 steps/epoch
#     (so the triangular period is 390 steps, not 180)
#   * evaluation on the full 10,000-image TEST split (no val holdout)
#   * per-channel std (0.2023, 0.1994, 0.2010) rather than (0.247, 0.243, 0.262)
# Use this backend when the goal is to reproduce or explain their numbers.
# It differs from `cifar` now only in the normalization constants and batch
# size; both train on the full 50k and evaluate on the 10k test split.
def _build_cifar_bench(cfg):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.models import resnet18

    mean = tuple(cfg.get("bench_mean", (0.4914, 0.4822, 0.4465)))
    std = tuple(cfg.get("bench_std", (0.2023, 0.1994, 0.2010)))

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean, std)])

    root = cfg["data_root"]
    ds = getattr(datasets, dataset)
    train_set = ds(root, train=True, download=True, transform=train_tf)
    test_set = ds(root, train=False, download=True, transform=eval_tf)

    # Reviewer 3.4: hyperparameters must not be selected on the test split.
    # With val_fraction > 0, hold out that fraction of the TRAINING data as a
    # validation set. Two dataset instances are built so the validation split gets
    # eval transforms (no random crop/flip) rather than train transforms.
    test_eval = None
    val_fraction = float(cfg.get("val_fraction", 0.0))
    if val_fraction > 0:
        import torch
        from torch.utils.data import Subset
        n_total = len(train_set)
        n_val = int(round(n_total * val_fraction))
        g = torch.Generator().manual_seed(int(cfg.get("split_seed", 0)))
        perm = torch.randperm(n_total, generator=g).tolist()
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        train_eval_ds = ds(root, train=True, download=True, transform=eval_tf)
        train_set = Subset(train_set, tr_idx)
        val_set = Subset(train_eval_ds, val_idx)

    bs = int(cfg["batch_size"])
    nw = cfg.get("num_workers", 4)
    # drop_last=True exactly as in jrnjf/sgd_jumper (50000/128 -> 390 steps)
    pw = nw > 0
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=pw)
    val_loader = DataLoader(test_set, batch_size=1024, shuffle=False,
                            num_workers=nw, pin_memory=True,
                            persistent_workers=pw)

    # No even-step requirement (the CyclicLR equivalence check is gone).
    steps = len(train_loader)

    def make_model():
        m = resnet18(num_classes=10)
        m.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        nn.init.kaiming_normal_(m.conv1.weight, mode="fan_out", nonlinearity="relu")
        return m

    device = cfg.get("device", "cuda")

    def _eval_on(model, loader):
        model.eval()
        crit = nn.CrossEntropyLoss()
        tot_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                tot_loss += crit(out, y).item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                n += x.size(0)
        return tot_loss / n, correct / n

    def eval_fn(model, _=None):
        return _eval_on(model, val_loader)

    return (train_loader, val_loader, test_loader,
            make_model, eval_fn, len(train_loader))


# ---------------------------------------------------------------------------
def build(cfg):
    name = cfg.get("backend", "tiny")
    if name == "tiny":
        return _build_tiny(cfg)
    if name == "cifar":
        return _build_cifar(cfg)
    if name == "cifar100":
        return _build_cifar100(cfg)
    if name == "cifar_bench":
        return _build_cifar_bench(cfg)
    raise SystemExit(f"unknown backend: {name!r} "
                     f"(expected 'tiny', 'cifar', 'cifar100' or 'cifar_bench')")
