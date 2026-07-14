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

    # Parse max_delta_mag (accepts int, list, or numeric string)
    max_delta_mag_raw = bulk["max_delta_mag"]
    if isinstance(max_delta_mag_raw, (int, float)):
        max_delta_mag = np.array([int(max_delta_mag_raw)], dtype=int)
    elif isinstance(max_delta_mag_raw, (list, tuple)):
        if not all(isinstance(v, (int, float)) for v in max_delta_mag_raw):
            raise ValueError("max_delta_mag list must contain only numeric values.")
        max_delta_mag = np.asarray([int(v) for v in max_delta_mag_raw], dtype=int)
    else:
        raise ValueError(f"Unsupported max_delta_mag type: {type(max_delta_mag_raw).__name__}")
    
    if np.any(max_delta_mag < 0):
        raise ValueError("BULK.SIMULATION.max_delta_mag must be >= 0")

    # Parse starting_magnitude (accepts float, list, or numeric string)
    start_mag_raw = bulk["starting_magnitude"]
    if isinstance(start_mag_raw, (int, float)):
        start_mag = np.array([float(start_mag_raw)], dtype=float)
    elif isinstance(start_mag_raw, (list, tuple)):
        if not all(isinstance(v, (int, float)) for v in start_mag_raw):
            raise ValueError("starting_magnitude list must contain only numeric values.")
        start_mag = np.asarray(start_mag_raw, dtype=float)
    else:
        raise ValueError(f"Unsupported starting_magnitude type: {type(start_mag_raw).__name__}")

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


def _parse_closed_range(range_value: Any, field: str) -> tuple[float, float]:
    if not isinstance(range_value, (list, tuple)) or len(range_value) != 2:
        raise ValueError(f"{field} must be a 2-element list [min, max].")
    if not all(isinstance(v, (int, float)) for v in range_value):
        raise ValueError(f"{field} bounds must be numeric.")

    lower = float(range_value[0])
    upper = float(range_value[1])
    if lower > upper:
        raise ValueError(f"{field} lower bound must be <= upper bound.")
    return lower, upper


def resolve_random_simulation_config(config_path: str | Path = "scripts/parameters.yaml") -> dict[str, Any]:
    cfg = _load_yaml(config_path)

    if "RANDOM.SIMULATION" not in cfg:
        raise KeyError("Missing RANDOM.SIMULATION section.")
    random_cfg = cfg["RANDOM.SIMULATION"] or {}

    for k in [
        "n_iter",
        "angle",
        "allow_duplicate_distances",
        "distance_range",
        "CENTRAL.RANDOMIZE",
        "COMPANION.RANDOMIZE",
    ]:
        if k not in random_cfg:
            raise KeyError(f"Missing RANDOM.SIMULATION.{k}")

    n_iter_raw = random_cfg["n_iter"]
    if not isinstance(n_iter_raw, (int, float)):
        raise ValueError("RANDOM.SIMULATION.n_iter must be numeric.")
    n_iter = int(n_iter_raw)
    if n_iter <= 0:
        raise ValueError("RANDOM.SIMULATION.n_iter must be > 0.")

    distance_min, distance_max = _parse_closed_range(
        random_cfg["distance_range"],
        "RANDOM.SIMULATION.distance_range",
    )

    central_cfg = random_cfg["CENTRAL.RANDOMIZE"] or {}
    companion_cfg = random_cfg["COMPANION.RANDOMIZE"] or {}
    for scope_name, scope in [
        ("CENTRAL.RANDOMIZE", central_cfg),
        ("COMPANION.RANDOMIZE", companion_cfg),
    ]:
        for k in ["temperature", "magnitude", "temperature_range", "magnitude_range"]:
            if k not in scope:
                raise KeyError(f"Missing RANDOM.SIMULATION.{scope_name}.{k}")

    band = random_cfg.get("band")
    if band is None:
        band = (cfg.get("BULK.SIMULATION") or {}).get("band")
    if band is None:
        band = (cfg.get("SINGLE.SIMULATION") or {}).get("band")
    if band is None:
        band = "gaia_g"

    return {
        "band": band,
        "n_iter": n_iter,
        "angle": float(random_cfg["angle"]) * u.deg,
        "shot_noise": _to_bool(
            random_cfg.get("shot_noise", False),
            "RANDOM.SIMULATION.shot_noise",
        ),
        "ron": _to_bool(
            random_cfg.get("ron", False),
            "RANDOM.SIMULATION.ron",
        ),
        "allow_duplicate_distances": _to_bool(
            random_cfg["allow_duplicate_distances"],
            "RANDOM.SIMULATION.allow_duplicate_distances",
        ),
        "distance_range": (distance_min, distance_max),
        "central_randomize": {
            "temperature": _to_bool(
                central_cfg["temperature"],
                "RANDOM.SIMULATION.CENTRAL.RANDOMIZE.temperature",
            ),
            "magnitude": _to_bool(
                central_cfg["magnitude"],
                "RANDOM.SIMULATION.CENTRAL.RANDOMIZE.magnitude",
            ),
            "temperature_range": _parse_closed_range(
                central_cfg["temperature_range"],
                "RANDOM.SIMULATION.CENTRAL.RANDOMIZE.temperature_range",
            ),
            "magnitude_range": _parse_closed_range(
                central_cfg["magnitude_range"],
                "RANDOM.SIMULATION.CENTRAL.RANDOMIZE.magnitude_range",
            ),
        },
        "companion_randomize": {
            "temperature": _to_bool(
                companion_cfg["temperature"],
                "RANDOM.SIMULATION.COMPANION.RANDOMIZE.temperature",
            ),
            "magnitude": _to_bool(
                companion_cfg["magnitude"],
                "RANDOM.SIMULATION.COMPANION.RANDOMIZE.magnitude",
            ),
            "temperature_range": _parse_closed_range(
                companion_cfg["temperature_range"],
                "RANDOM.SIMULATION.COMPANION.RANDOMIZE.temperature_range",
            ),
            "magnitude_range": _parse_closed_range(
                companion_cfg["magnitude_range"],
                "RANDOM.SIMULATION.COMPANION.RANDOMIZE.magnitude_range",
            ),
        },
        "raw": random_cfg,
    }


def resolve_simulation_config(mode: str, config_path: str | Path = "scripts/parameters.yaml") -> dict[str, Any]:
    mode_norm = mode.strip().lower()
    if mode_norm == "bulk":
        return resolve_bulk_simulation_config(config_path)
    if mode_norm == "single":
        return resolve_single_simulation_config(config_path)
    if mode_norm == "random":
        return resolve_random_simulation_config(config_path)
    raise ValueError("mode must be 'bulk', 'single' or 'random'.")