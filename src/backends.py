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

    steps = len(train_full) // bs
    if steps % 2:
        raise SystemExit(
            f"tiny backend: {len(train_full)} samples / batch {bs} = {steps} steps, "
            f"which is ODD. The triangular half-cycle would be {steps/2}. "
            f"Pick a batch size that divides to an even count "
            f"(e.g. batch_size={len(train_full)//(steps-1) if steps>2 else bs})."
        )

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

    def eval_fn(model, _=None):
        model.eval()
        crit = nn.CrossEntropyLoss()
        tot_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                tot_loss += crit(out, y).item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                n += x.size(0)
        return tot_loss / n, correct / n

    return (train_loader, val_loader, test_loader,
            make_model, eval_fn, len(train_loader))


# ---------------------------------------------------------------------------
# cifar: CIFAR-10 + ResNet-18, GPU
# ---------------------------------------------------------------------------
# Trains on the FULL 50,000-image training set and evaluates on the official
# 10,000-image TEST split. There is deliberately no validation holdout: carving
# one out of train both discards training data and changes steps_per_epoch,
# which in turn changes the triangular period (the schedule depends on it).
def _build_cifar(cfg):
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
    train_set = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)

    bs = int(cfg["batch_size"])
    nw = cfg.get("num_workers", 4)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    # test set doubles as the evaluation set
    eval_loader = DataLoader(test_set, batch_size=512, shuffle=False,
                             num_workers=nw, pin_memory=True)

    steps = len(train_loader)          # e.g. 50000/250 = 200
    if steps % 2:
        raise SystemExit(
            f"cifar: len(train_loader) = {steps} is ODD, so the triangular "
            f"half-cycle is {steps/2} and no CyclicLR configuration reproduces "
            f"the internal wave exactly. With 50000 train images, use a batch "
            f"size giving an even step count (250 -> 200, 200 -> 250, 100 -> 500)."
        )

    def make_model():
        m = resnet18(num_classes=10)
        m.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m

    device = cfg.get("device", "cuda")

    def eval_fn(model, _=None):
        model.eval()
        crit = nn.CrossEntropyLoss()
        tot_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                tot_loss += crit(out, y).item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                n += x.size(0)
        return tot_loss / n, correct / n

    return (train_loader, eval_loader, eval_loader, make_model, eval_fn,
            len(train_loader))


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
    train_set = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)

    bs = int(cfg["batch_size"])
    nw = cfg.get("num_workers", 4)
    # drop_last=True exactly as in jrnjf/sgd_jumper (50000/128 -> 390 steps)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(test_set, batch_size=512, shuffle=False,
                            num_workers=nw, pin_memory=True)

    steps = len(train_loader)
    if steps % 2:
        raise SystemExit(f"cifar_bench: len(train_loader)={steps} is ODD; the "
                         f"triangular half-cycle {steps/2} cannot be reproduced.")

    def make_model():
        m = resnet18(num_classes=10)
        m.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        nn.init.kaiming_normal_(m.conv1.weight, mode="fan_out", nonlinearity="relu")
        return m

    device = cfg.get("device", "cuda")

    def eval_fn(model, _=None):
        model.eval()
        crit = nn.CrossEntropyLoss()
        tot_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                tot_loss += crit(out, y).item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                n += x.size(0)
        return tot_loss / n, correct / n

    return (train_loader, val_loader, test_loader, make_model, eval_fn,
            len(train_loader))


# ---------------------------------------------------------------------------
def build(cfg):
    name = cfg.get("backend", "tiny")
    if name == "tiny":
        return _build_tiny(cfg)
    if name == "cifar":
        return _build_cifar(cfg)
    if name == "cifar_bench":
        return _build_cifar_bench(cfg)
    raise SystemExit(f"unknown backend: {name!r} "
                     f"(expected 'tiny', 'cifar' or 'cifar_bench')")
