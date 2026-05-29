from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import numpy as np
import yaml
from astropy import units as u


def _to_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field} must be boolean (true/false), got {type(value).__name__}.")


def _parse_distance(distance_value: Any) -> np.ndarray:
    # Supported:
    # - 150
    # - [20, 40, 60]
    # - "np.arange(20,220,10)"
    if isinstance(distance_value, (int, float)):
        return np.array([float(distance_value)], dtype=float)

    if isinstance(distance_value, (list, tuple)):
        if not all(isinstance(v, (int, float)) for v in distance_value):
            raise ValueError("distance list must contain only numeric values.")
        return np.asarray(distance_value, dtype=float)

    if isinstance(distance_value, str):
        expr = distance_value.replace(" ", "")
        m = re.fullmatch(
            r"np\.arange\(([-+]?\d*\.?\d+),([-+]?\d*\.?\d+)(?:,([-+]?\d*\.?\d+))?\)",
            expr,
        )
        if not m:
            raise ValueError(
                "distance string must be np.arange(start,stop[,step]) or numeric/list."
            )
        start = float(m.group(1))
        stop = float(m.group(2))
        step = float(m.group(3)) if m.group(3) is not None else 1.0
        if step == 0:
            raise ValueError("np.arange step cannot be zero.")
        return np.arange(start, stop, step, dtype=float)

    raise ValueError(f"Unsupported distance type: {type(distance_value).__name__}")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Top-level YAML content must be a mapping.")
    return cfg


def resolve_bulk_simulation_config(config_path: str | Path = "scripts/parameters.yaml") -> dict[str, Any]:
    cfg = _load_yaml(config_path)

    if "BULK.SIMULATION" not in cfg:
        raise KeyError("Missing BULK.SIMULATION section.")
    bulk = cfg["BULK.SIMULATION"] or {}

    required = [
        "band",
        "max_delta_mag",
        "starting_magnitude",
        "central_star_temperature",
        "companion_star_temperature",
        "distance",
        "angle",
        "shot_noise",
        "ron",
    ]
    for k in required:
        if k not in bulk:
            raise KeyError(f"Missing BULK.SIMULATION.{k}")

    max_delta_mag = int(bulk["max_delta_mag"])
    if max_delta_mag < 0:
        raise ValueError("BULK.SIMULATION.max_delta_mag must be >= 0")

    start_mag = float(bulk["starting_magnitude"])
    central_temp = float(bulk["central_star_temperature"])
    companion_temp = float(bulk["companion_star_temperature"])

    distances = _parse_distance(bulk["distance"]) * u.mas
    angle = float(bulk["angle"]) * u.deg
    shot_noise = _to_bool(bulk["shot_noise"], "BULK.SIMULATION.shot_noise")
    ron = _to_bool(bulk["ron"], "BULK.SIMULATION.ron")

    return {
        'band': bulk["band"],
        'delta_mag': max_delta_mag,
        'starting_magnitude': start_mag,
        "distances": distances,
        "angle": angle,
        "shot_noise": shot_noise,
        "ron": ron,
        "central_star_temperature": central_temp,
        "companion_star_temperature": companion_temp,
    }


def resolve_single_simulation_config(config_path: str | Path = "scripts/parameters.yaml") -> dict[str, Any]:
    cfg = _load_yaml(config_path)

    if "SINGLE.SIMULATION" not in cfg:
        raise KeyError("Missing SINGLE.SIMULATION section.")
    single = cfg["SINGLE.SIMULATION"] or {}

    for k in ["distance", "angle", "shot_noise", "ron", "CENTRAL.STAR", "COMPANION.STAR"]:
        if k not in single:
            raise KeyError(f"Missing SINGLE.SIMULATION.{k}")

    central = single["CENTRAL.STAR"] or {}
    companion = single["COMPANION.STAR"] or {}

    for k in ["temperature", "magnitude"]:
        if k not in central:
            raise KeyError(f"Missing SINGLE.SIMULATION.CENTRAL.STAR.{k}")
        if k not in companion:
            raise KeyError(f"Missing SINGLE.SIMULATION.COMPANION.STAR.{k}")

    dvals = _parse_distance(single["distance"])
    if dvals.size != 1:
        raise ValueError("SINGLE.SIMULATION.distance must resolve to exactly one value.")
    distance = float(dvals[0]) * u.mas

    return {
        "band": single.get("band"),
        "M1": float(central["magnitude"]),
        "M2": float(companion["magnitude"]),
        "distance": distance,
        "angle": float(single["angle"]) * u.deg,
        "shot_noise": _to_bool(single["shot_noise"], "SINGLE.SIMULATION.shot_noise"),
        "ron": _to_bool(single["ron"], "SINGLE.SIMULATION.ron"),
        "central_star_temperature": float(central["temperature"]),
        "companion_star_temperature": float(companion["temperature"]),
        "central_spectral_type": central.get("spectral_type"),
        "companion_spectral_type": companion.get("spectral_type"),
        "raw": single,
    }


def resolve_simulation_config(mode: str, config_path: str | Path = "scripts/parameters.yaml") -> dict[str, Any]:
    mode_norm = mode.strip().lower()
    if mode_norm == "bulk":
        return resolve_bulk_simulation_config(config_path)
    if mode_norm == "single":
        return resolve_single_simulation_config(config_path)
    raise ValueError("mode must be 'bulk' or 'single'.")