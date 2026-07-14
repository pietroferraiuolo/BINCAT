import gc
import os

import numpy as np
import xupy as xp
import bincatsim as bs
from astropy import units as u
from bincatsim.utils.read_config import resolve_random_simulation_config


try:
    xp.set_device(1)  # Set to 2nd GPU (gandalf)
except Exception:
    pass


ccd = bs.CCD(
    psf=bs.paths.PSF_FILE,
)


def analyze_simulation(tn: str) -> dict[str, float | None]:
    ipd = bs.IPD(tn)
    ipd(epsilon=1e-3)
    return {
        "gof_amp": ipd.gof_amp,
        "gof_phase": ipd.gof_phase,
        "al_multipeak": ipd.frac_multipeak,
        "ac_multipeak": 0.0,
        "delta_m": None,
        "frac_badfit": ipd.frac_badfit,
        "chi2_threshold": ipd._chi2_threshold,
        "phi_threshold": ipd._phi_threshold,
    }


def _resolve_random_value(
    rng: np.random.Generator,
    bounds: tuple[float, float],
    randomize: bool,
) -> float:
    lower, upper = bounds
    if randomize:
        return float(rng.uniform(lower, upper))
    return float((lower + upper) / 2.0)


def _sample_distances(
    rng: np.random.Generator,
    n_iter: int,
    distance_range: tuple[float, float],
    allow_duplicate_distances: bool,
) -> np.ndarray:
    lower, upper = distance_range

    if np.isclose(lower, upper):
        return np.full(n_iter, float(lower), dtype=float)

    integer_low = int(np.ceil(lower))
    integer_high = int(np.floor(upper))

    if integer_low <= integer_high:
        population = np.arange(integer_low, integer_high + 1, dtype=float)
        if allow_duplicate_distances:
            idx = rng.integers(0, population.size, size=n_iter)
            return population[idx]

        if n_iter > population.size:
            raise ValueError(
                "RANDOM.SIMULATION.n_iter exceeds number of available integer distances "
                "for the configured distance_range when allow_duplicate_distances is false."
            )
        return rng.choice(population, size=n_iter, replace=False)

    if allow_duplicate_distances:
        return rng.uniform(lower, upper, size=n_iter)

    distances = np.linspace(lower, upper, n_iter, endpoint=True)
    rng.shuffle(distances)
    return distances


if __name__ == "__main__":
    parfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.yaml")
    config = resolve_random_simulation_config(parfile)

    n_iter = config["n_iter"]
    angle = config["angle"]
    band = config["band"]
    shot_noise = config["shot_noise"]
    ron = config["ron"]

    central_cfg = config["central_randomize"]
    companion_cfg = config["companion_randomize"]

    rng = np.random.default_rng()
    sampled_distances = _sample_distances(
        rng=rng,
        n_iter=n_iter,
        distance_range=config["distance_range"],
        allow_duplicate_distances=config["allow_duplicate_distances"],
    )

    tn: str | None = None

    try:
        print(f"Starting random simulation: {n_iter} simulations to run.\n")

        for k in range(n_iter):
            c_mag = _resolve_random_value(
                rng,
                central_cfg["magnitude_range"],
                central_cfg["magnitude"],
            )
            s_mag = _resolve_random_value(
                rng,
                companion_cfg["magnitude_range"],
                companion_cfg["magnitude"],
            )
            c_temp = _resolve_random_value(
                rng,
                central_cfg["temperature_range"],
                central_cfg["temperature"],
            )
            s_temp = _resolve_random_value(
                rng,
                companion_cfg["temperature_range"],
                companion_cfg["temperature"],
            )
            distance = float(sampled_distances[k]) * u.mas

            print(
                f"[{k + 1}/{n_iter}] : Distance={distance}; "
                f"central=(mag={c_mag:.3f}, T={c_temp:.1f} K), "
                f"companion=(mag={s_mag:.3f}, T={s_temp:.1f} K)",
                end="\n",
            )

            central_star = bs.Star(
                magnitude=c_mag,
                type_or_temp=c_temp,
                band=band,
            )
            companion_star = bs.Star(
                magnitude=s_mag,
                type_or_temp=s_temp,
                band=band,
            )

            sim = bs.GaiaSimulator(
                ccd=ccd,
                central_star=central_star,
                companion_star=companion_star,
                distance=distance,
                angle=angle,
            )

            tn = sim.observe(
                shot_noise=shot_noise,
                read_out_noise=ron,
            )

            resdict = analyze_simulation(tn)
            resdict["delta_m"] = s_mag - c_mag
            sim.update_record_file(tn, other_params=resdict)

            del sim
            gc.collect()

    except KeyboardInterrupt:
        from bincatsim.core.root import OBS_DATA_PATH
        import shutil

        if tn is not None:
            fold_to_delete = os.path.join(OBS_DATA_PATH, tn)
            if os.path.exists(fold_to_delete):
                shutil.rmtree(fold_to_delete)
                print(f"Deleted incomplete simulation folder: {fold_to_delete}")
        print("Simulation interrupted.")