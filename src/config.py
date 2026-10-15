"""Tiny YAML config loader with command-line overrides.

Usage:
    cfg = load_config("configs/x.yaml", ["train.epochs=2", "model.name=foo"])
    cfg["train"]["epochs"]  -> 2
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


def _coerce(value: str) -> Any:
    """Turn a CLI string into int/float/bool/None/list when it looks like one."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(value)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return apply_overrides(cfg, overrides)


def base_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config, help="YAML config file")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set train.epochs=1 data.max_train=2000",
    )
    return parser
