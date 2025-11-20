"""Tiny resnet50 fine-tune stub. Produces a TorchScript artefact + .mar
ready for torchserve.

Real training would pull from imagenet/CIFAR; this script just runs a
short loop on synthetic tensors so we have a non-empty model.pt that
the runtime can load. The point of this repo is the serving stack, not
training a SOTA classifier.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/resnet50")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    net: nn.Module = models.resnet50(weights=None, num_classes=10).to(device)
    opt = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    crit = nn.CrossEntropyLoss()

    net.train()
    for step in range(args.steps):
        x = torch.randn(16, 3, 224, 224, device=device)
        y = torch.randint(0, 10, (16,), device=device)
        opt.zero_grad()
        logits = net(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        if step % 5 == 0:
            print(f"step {step:>3} loss={loss.item():.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net.eval()
    scripted = torch.jit.script(net.cpu())
    scripted.save(out / "model.pt")
    print(f"wrote {out / 'model.pt'}")


if __name__ == "__main__":
    main()
