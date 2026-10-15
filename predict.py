"""Run a trained checkpoint on new data and write georeferenced masks.

Landslide4Sense patch (.h5):
    python predict.py --ckpt outputs/l4s-unet-r34/best.pt --h5 image_12.h5 --out pred.png

Any-size multi-band GeoTIFF (same band order as training):
    python predict.py --ckpt outputs/l4s-unet-r34/best.pt --tif scene.tif --out scene_mask.tif

Bi-temporal SAR pair (pairs mode):
    python predict.py --ckpt outputs/sar-change-unet-r34/best.pt --pre pre.tif --post post.tif \
        --out change_mask.tif

GeoTIFF outputs have 2 bands: 1 = probability (0-1), 2 = binary mask. CRS and transform
are copied from the input, so they open straight in QGIS.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import Normaliser, read_h5, read_tif, sar_features, to_chw
from src.model import build_model


@torch.no_grad()
def predict_tiled(model, x: np.ndarray, device, tile: int = 256, overlap: int = 64) -> np.ndarray:
    """Sliding-window inference with cosine blending. x: (C,H,W) normalised -> (H,W) probs."""
    c, h, w = x.shape
    # pad to a multiple of 32 (encoder stride); small images become a single window
    H, W = h + (-h) % 32, w + (-w) % 32
    can_reflect = (H - h) < h and (W - w) < w  # reflect padding must be smaller than the image
    xp = np.pad(x, ((0, 0), (0, H - h), (0, W - w)), mode="reflect" if can_reflect else "constant")
    t = min(tile, H, W)
    stride = max(1, t - overlap)
    win = np.outer(np.hanning(t), np.hanning(t)).astype(np.float32) + 1e-3
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    tops = list(range(0, max(H - t, 0) + 1, stride))
    lefts = list(range(0, max(W - t, 0) + 1, stride))
    if tops[-1] != H - t:
        tops.append(H - t)
    if lefts[-1] != W - t:
        lefts.append(W - t)
    for top in tops:
        for left in lefts:
            patch = torch.from_numpy(np.ascontiguousarray(xp[:, top : top + t, left : left + t]))[None].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prob = torch.sigmoid(model(patch).float())[0, 0].cpu().numpy()
            acc[top : top + t, left : left + t] += prob * win
            wsum[top : top + t, left : left + t] += win
    return (acc / wsum)[:h, :w]


def write_geotiff(ref_path: Path, out_path: Path, prob: np.ndarray, threshold: float) -> None:
    import rasterio

    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
    profile.update(count=2, dtype="float32", nodata=None, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(prob.astype(np.float32), 1)
        dst.write((prob >= threshold).astype(np.float32), 2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--h5")
    p.add_argument("--tif")
    p.add_argument("--pre")
    p.add_argument("--post")
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--tile", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model({**ck["model_cfg"], "encoder_weights": None}, ck["in_channels"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    norm = Normaliser.load(Path(args.ckpt).parent / "band_stats.json")
    dcfg = ck["data_cfg"]
    thr = args.threshold if args.threshold is not None else 0.5

    if args.pre and args.post:
        x = sar_features(read_tif(Path(args.pre)), read_tif(Path(args.post)), dcfg.get("sar", True))
        ref = Path(args.post)
    elif args.tif:
        x = read_tif(Path(args.tif))
        bands = dcfg.get("bands", "all")
        if bands != "all":
            x = x[list(bands)]
        ref = Path(args.tif)
    elif args.h5:
        x = to_chw(read_h5(Path(args.h5), ("img", "image", "data")))
        bands = dcfg.get("bands", "all")
        if bands != "all":
            x = x[list(bands)]
        ref = None
    else:
        raise SystemExit("Give --h5, --tif, or --pre and --post")

    if x.shape[0] != ck["in_channels"]:
        raise SystemExit(f"Input has {x.shape[0]} channels, model expects {ck['in_channels']}")
    prob = predict_tiled(model, norm(x), device, tile=args.tile)
    out = Path(args.out)
    if ref is not None and out.suffix.lower() in (".tif", ".tiff"):
        write_geotiff(ref, out, prob, thr)
    else:
        import matplotlib.pyplot as plt

        plt.imsave(out, prob >= thr, cmap="gray")
    print(f"changed pixels: {(prob >= thr).mean() * 100:.2f}%  ->  {out}")


if __name__ == "__main__":
    main()
