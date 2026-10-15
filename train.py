"""Fine-tune a pretrained segmentation network for landslide / SAR change mapping.

    python train.py --config configs/landslide4sense.yaml
    python train.py --config configs/sar_change_pairs.yaml
    python train.py --set train.epochs=3                # smoke run
    python train.py --set model.encoder=efficientnet-b3 train.output_dir=outputs/l4s-effb3
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from src.config import base_parser, load_config
from src.data import build_datasets
from src.metrics import BinaryStats
from src.model import build_model, param_groups, set_encoder_trainable


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, loss_fn, device, threshold, optimizer=None, scaler=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    stats, total, n = BinaryStats(threshold), 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(x)
            loss = loss_fn(logits.float(), y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        stats.update(logits, y)
        total += float(loss) * x.size(0)
        n += x.size(0)
    return total / max(n, 1), stats.compute()


def main() -> None:
    args = base_parser(__doc__, "configs/landslide4sense.yaml").parse_args()
    cfg = load_config(args.config, args.set)
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    seed_all(tcfg["seed"])
    out = Path(tcfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, in_ch, _ = build_datasets(dcfg, out, tcfg["seed"])
    print(f"train={len(train_ds)} val={len(val_ds)} in_channels={in_ch}")
    loader_kw = {"batch_size": tcfg["batch_size"], "num_workers": tcfg["num_workers"],
                 "pin_memory": device.type == "cuda"}
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = build_model(mcfg, in_ch).to(device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([tcfg["pos_weight"]], device=device))
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def loss_fn(logits, y):
        return bce(logits, y) + dice(logits, y)

    optimizer = torch.optim.AdamW(
        param_groups(model, tcfg["lr"], tcfg["encoder_lr_mult"], tcfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in optimizer.param_groups],
        total_steps=tcfg["epochs"] * len(train_dl),
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_iou, history = -1.0, []
    for epoch in range(1, tcfg["epochs"] + 1):
        frozen = epoch <= tcfg["freeze_encoder_epochs"]
        set_encoder_trainable(model, not frozen)
        t0 = time.time()
        tr_loss, tr_m = run_epoch(
            model, train_dl, loss_fn, device, tcfg["threshold"], optimizer, scaler, scheduler
        )
        va_loss, va_m = run_epoch(model, val_dl, loss_fn, device, tcfg["threshold"])
        row = {"epoch": epoch, "encoder_frozen": frozen, "train_loss": round(tr_loss, 4),
               "val_loss": round(va_loss, 4), **{f"val_{k}": round(v, 4) for k, v in va_m.items()},
               "train_iou": round(tr_m["iou"], 4), "sec": round(time.time() - t0, 1)}
        history.append(row)
        print(json.dumps(row))
        if va_m["iou"] > best_iou:
            best_iou = va_m["iou"]
            torch.save({"state_dict": model.state_dict(), "model_cfg": mcfg, "in_channels": in_ch,
                        "data_cfg": dcfg, "epoch": epoch, "val": va_m}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    best = max(history, key=lambda r: r["val_iou"])
    print(f"Best val IoU {best['val_iou']:.4f}, F1 {best['val_f1']:.4f} at epoch {best['epoch']}")
    print(f"Checkpoint: {out / 'best.pt'}")


if __name__ == "__main__":
    main()
