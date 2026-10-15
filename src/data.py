"""Datasets: Landslide4Sense (.h5 patches) and bi-temporal GeoTIFF pairs."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ----------------------------------------------------------------- helpers


def read_h5(path: Path, preferred: tuple[str, ...]) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        key = next((k for k in preferred if k in f), None) or next(iter(f.keys()))
        return np.asarray(f[key][()])


def to_chw(arr: np.ndarray) -> np.ndarray:
    """Landslide4Sense stores HWC; make it CHW."""
    if arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
        arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)


def sar_features(pre: np.ndarray, post: np.ndarray, is_sar: bool) -> np.ndarray:
    """[pre, post, change] stacked on channels. SAR is converted to dB and the change
    channel is the log-ratio (difference in dB), the standard SAR change indicator."""
    if is_sar:
        pre_db = 10 * np.log10(np.clip(pre, 1e-6, None))
        post_db = 10 * np.log10(np.clip(post, 1e-6, None))
        return np.concatenate([pre_db, post_db, post_db - pre_db], 0).astype(np.float32)
    return np.concatenate([pre, post, post - pre], 0).astype(np.float32)


def augment(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randint(0, 3)
    x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(0, 1))
    if random.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    if random.random() < 0.5:
        x, y = x[:, ::-1, :], y[::-1, :]
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


class Normaliser:
    def __init__(self, mean: list[float], std: list[float]):
        self.mean = np.asarray(mean, np.float32)[:, None, None]
        self.std = np.asarray(std, np.float32)[:, None, None]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = (x - self.mean) / self.std
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"mean": self.mean.ravel().tolist(), "std": self.std.ravel().tolist()}))

    @classmethod
    def load(cls, path: Path) -> "Normaliser":
        d = json.loads(path.read_text())
        return cls(d["mean"], d["std"])

    @classmethod
    def fit(cls, samples: list[np.ndarray]) -> "Normaliser":
        stack = np.stack([s.reshape(s.shape[0], -1) for s in samples], 0)  # N,C,P
        stack = np.transpose(stack, (1, 0, 2)).reshape(stack.shape[1], -1)
        stack = np.where(np.isfinite(stack), stack, np.nan)
        mean = np.nanmean(stack, 1)
        std = np.nanstd(stack, 1)
        std[std < 1e-6] = 1.0
        return cls(mean.tolist(), std.tolist())


# ----------------------------------------------------------------- Landslide4Sense


def download_landslide4sense(repo: str, root: Path) -> None:
    if (root / "images" / "train").exists():
        return
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo} -> {root} (a few GB, first time only) ...")
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=["images/train/*", "images/validation/*", "annotations/train/*",
                        "annotations/validation/*", "images/test/*"],
    )


def l4s_pairs(root: Path, split: str) -> list[tuple[Path, Path | None]]:
    img_dir, ann_dir = root / "images" / split, root / "annotations" / split
    out = []
    for img in sorted(img_dir.glob("image_*.h5"), key=lambda p: int(p.stem.split("_")[1])):
        mask = ann_dir / img.name.replace("image_", "mask_")
        out.append((img, mask if mask.exists() else None))
    return out


class Landslide4SenseDataset(Dataset):
    def __init__(self, items, bands, norm: Normaliser | None, train: bool):
        self.items, self.bands, self.norm, self.train = items, bands, norm, train

    def load_image(self, path: Path) -> np.ndarray:
        x = to_chw(read_h5(path, ("img", "image", "data")))
        return x[self.bands] if self.bands is not None else x

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        img_path, mask_path = self.items[i]
        x = self.load_image(img_path)
        y = (read_h5(mask_path, ("mask", "label")).astype(np.float32) > 0).astype(np.float32)
        if self.train:
            x, y = augment(x, y)
        if self.norm:
            x = self.norm(x)
        return torch.from_numpy(x), torch.from_numpy(y)[None]


# ----------------------------------------------------------------- GeoTIFF pairs


def read_tif(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


class PairsDataset(Dataset):
    """Random tiles from bi-temporal scenes. Each item -> (3C,H,W) features, (1,H,W) mask."""

    def __init__(self, scene_dirs, tile, tiles_per_scene, is_sar, norm, train):
        self.tile, self.is_sar, self.norm, self.train = tile, is_sar, norm, train
        self.tiles_per_scene = tiles_per_scene if train else 1
        self.scenes = []
        for d in scene_dirs:
            feats = sar_features(read_tif(d / "pre.tif"), read_tif(d / "post.tif"), is_sar)
            mask = (read_tif(d / "mask.tif")[0] > 0).astype(np.float32)
            self.scenes.append((feats, mask))

    def __len__(self) -> int:
        return len(self.scenes) * self.tiles_per_scene

    def _crop(self, feats, mask):
        _, h, w = feats.shape
        t = self.tile
        if h < t or w < t:  # pad small scenes
            ph, pw = max(0, t - h), max(0, t - w)
            feats = np.pad(feats, ((0, 0), (0, ph), (0, pw)))
            mask = np.pad(mask, ((0, ph), (0, pw)))
            _, h, w = feats.shape
        if self.train:
            # bias crops toward changed areas half of the time
            ys, xs = np.nonzero(mask)
            if len(ys) and random.random() < 0.5:
                j = random.randrange(len(ys))
                top = int(np.clip(ys[j] - t // 2, 0, h - t))
                left = int(np.clip(xs[j] - t // 2, 0, w - t))
            else:
                top, left = random.randint(0, h - t), random.randint(0, w - t)
        else:
            top, left = (h - t) // 2, (w - t) // 2
        return feats[:, top : top + t, left : left + t], mask[top : top + t, left : left + t]

    def __getitem__(self, i):
        feats, mask = self.scenes[i % len(self.scenes)]
        x, y = self._crop(feats, mask)
        if self.train:
            x, y = augment(x, y)
        if self.norm:
            x = self.norm(x)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))[None]


# ----------------------------------------------------------------- factory


def build_datasets(dcfg: dict, out_dir: Path, seed: int):
    """Returns (train_ds, val_ds, in_channels, normaliser). Saves band stats to out_dir."""
    rng = random.Random(seed)
    root = Path(dcfg["root"])

    if dcfg["mode"] == "landslide4sense":
        download_landslide4sense(dcfg["hf_repo"], root)
        train_items = [it for it in l4s_pairs(root, "train") if it[1] is not None]
        val_items = [it for it in l4s_pairs(root, "validation") if it[1] is not None]
        if dcfg.get("val_fraction", 0) > 0 or not val_items:
            rng.shuffle(train_items)
            n_val = max(1, int(len(train_items) * (dcfg.get("val_fraction") or 0.1)))
            val_items, train_items = train_items[:n_val], train_items[n_val:]
        bands = None if dcfg["bands"] == "all" else list(dcfg["bands"])
        probe = Landslide4SenseDataset(train_items, bands, None, train=False)
        samples = [probe.load_image(p) for p, _ in rng.sample(train_items, min(dcfg["stats_samples"], len(train_items)))]
        norm = Normaliser.fit(samples)
        train_ds = Landslide4SenseDataset(train_items, bands, norm, train=True)
        val_ds = Landslide4SenseDataset(val_items, bands, norm, train=False)
        in_ch = samples[0].shape[0]

    elif dcfg["mode"] == "pairs":
        tr_dirs = sorted(p for p in (root / "train").iterdir() if p.is_dir())
        va_root = root / "val"
        va_dirs = sorted(p for p in va_root.iterdir() if p.is_dir()) if va_root.exists() else []
        if not va_dirs:
            rng.shuffle(tr_dirs)
            n_val = max(1, int(len(tr_dirs) * (dcfg.get("val_fraction") or 0.15)))
            va_dirs, tr_dirs = tr_dirs[:n_val], tr_dirs[n_val:]
        train_ds = PairsDataset(tr_dirs, dcfg["tile"], dcfg["tiles_per_scene"], dcfg["sar"], None, True)
        samples = [train_ds[i][0].numpy() for i in range(min(dcfg["stats_samples"], len(train_ds)))]
        norm = Normaliser.fit(samples)
        train_ds.norm = norm
        val_ds = PairsDataset(va_dirs, dcfg["tile"], 1, dcfg["sar"], norm, False)
        in_ch = samples[0].shape[0]
    else:
        raise ValueError(f"Unknown data.mode {dcfg['mode']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    norm.save(out_dir / "band_stats.json")
    return train_ds, val_ds, in_ch, norm
