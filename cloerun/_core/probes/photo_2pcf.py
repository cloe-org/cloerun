"""Wrapper for the real-space photometric likelihoods of
`cloelike.EuclidLikelihood_photo_2pcf`: ``WL_2pcf``, ``GCph_2pcf``,
``GGL_2pcf``, ``2x2pt_2pcf`` and ``3x2pt_2pcf``.
"""

from __future__ import annotations

from typing import Any

import euclidlib as el
import numpy as np
from cloelike.EuclidLikelihood_photo_2pcf import (
    EuclidLikelihood_2x2pt,
    EuclidLikelihood_3x2pt,
    EuclidLikelihood_GCph,
    EuclidLikelihood_GGL,
    EuclidLikelihood_WL,
)

from cloerun._core.config import PHOTO_BLOCKS, RunConfig
from cloerun._core.loader import (
    load_background,
    load_nonlinear_perturbations,
    load_perturbations,
)
from cloerun._core.probes.photo_cls import (  # noqa: F401
    _build_settings,
    _load_covariance,
    _load_nz,
    derived_outputs,
    loglike,
    obs_labels,
    parameter_names,
    set_data_vector_masked,
    tracked_packages,
    use_noise_free_mock,
    vector_parameter_names,
)

_LIKELIHOODS = {
    "WL_2pcf": EuclidLikelihood_WL,
    "GCph_2pcf": EuclidLikelihood_GCph,
    "GGL_2pcf": EuclidLikelihood_GGL,
    "2x2pt_2pcf": EuclidLikelihood_2x2pt,
    "3x2pt_2pcf": EuclidLikelihood_3x2pt,
}

_BLOCKS = {
    "WL_2pcf": ("WL",),
    "GCph_2pcf": ("GCph",),
    "GGL_2pcf": ("GGL",),
    "2x2pt_2pcf": ("GGL", "GCph"),
    "3x2pt_2pcf": ("WL", "GGL", "GCph"),
}

_STATISTICS = {"WL": ("xi_plus", "xi_minus"), "GGL": ("gamma_t",), "GCph": ("w",)}

DEFAULT_ELL_MAX_INTEGRATION = 40000


def binning_theta(spec: dict[str, Any]) -> np.ndarray:
    """Centres in arcmin of ``N_theta`` log bins from ``theta_min`` to ``theta_max``."""
    edges = np.geomspace(spec["theta_min"], spec["theta_max"], int(spec["N_theta"]) + 1)
    return np.sqrt(edges[:-1] * edges[1:])


def _pairs(block: str, n_pos: int, n_she: int) -> list[tuple[int, int]]:
    """Tomographic pairs of a block, in the order of `cloelike`."""
    if block == "WL":
        return [(i, j) for i in range(1, n_she + 1) for j in range(i, n_she + 1)]
    if block == "GGL":
        return [(i, j) for i in range(1, n_pos + 1) for j in range(1, n_she + 1)]
    return [(i, j) for i in range(1, n_pos + 1) for j in range(i, n_pos + 1)]


def _data_keys(kind: str, n_pos: int, n_she: int) -> list[tuple]:
    """Keys of the correlation functions of the probe."""
    return [
        (*PHOTO_BLOCKS[block], i, j)
        for block in _BLOCKS[kind]
        for i, j in _pairs(block, n_pos, n_she)
    ]


def _template_twopcf(kind: str, n_pos: int, n_she: int, n_theta: int) -> dict[tuple, np.ndarray]:
    """Zero correlation functions with the keys and shapes `cloelike` reads."""
    shapes = {
        ("SHE", "SHE"): (2, 2, n_theta), ("POS", "SHE"): (2, n_theta), ("POS", "POS"): (n_theta,)
    }
    return {key: np.zeros(shapes[key[:2]]) for key in _data_keys(kind, n_pos, n_she)}


def _block_sizes(n_pos: int, n_she: int, n_theta: int) -> dict[str, int]:
    """Length of each statistic of the data vector."""
    n_wl = n_she * (n_she + 1) // 2 * n_theta
    return {
        "xi_plus": n_wl,
        "xi_minus": n_wl,
        "gamma_t": n_pos * n_she * n_theta,
        "w": n_pos * (n_pos + 1) // 2 * n_theta,
    }


def _statistics(kind: str) -> list[str]:
    """Statistics of the probe's data vector, in order."""
    return [stat for block in _BLOCKS[kind] for stat in _STATISTICS[block]]


def _theta_range(bounds: Any, block: str, label: str) -> list[float]:
    """Theta range of one pair: two numbers, or four for WL (xi_+ then xi_-)."""
    if isinstance(bounds, dict):
        if block != "WL" or set(bounds) != {"plus", "minus"}:
            raise ValueError(f"'scale_cuts' of {label}: expected {{plus: [..], minus: [..]}}")
        bounds = [*bounds["plus"], *bounds["minus"]]
    bounds = [float(b) for b in bounds]
    expected = 4 if block == "WL" else 2
    if len(bounds) != expected:
        raise ValueError(
            f"'scale_cuts' of {label}: expected {expected} numbers "
            f"({'xi_+ min, max, xi_- min, max' if block == 'WL' else 'theta min, max'}), got {bounds}"
        )
    return bounds


def _build_scale_cuts(cfg: RunConfig, twopcf: dict) -> dict[tuple, list[float]]:
    """Theta range of every correlation function of the probe."""
    kind = cfg.probe["kind"]
    missing = [block for block in _BLOCKS[kind] if block not in cfg.scale_cuts]
    if missing:
        raise ValueError(f"'scale_cuts' is missing the range of block(s) {missing}")
    prefixes = {PHOTO_BLOCKS[block]: block for block in _BLOCKS[kind]}
    overrides = cfg.scale_cuts["overrides"]
    unknown = [key for key in overrides if key not in twopcf]
    if unknown:
        raise ValueError(f"'scale_cuts.overrides' has pairs not in the data: {unknown}")
    scale_cuts = {}
    for key in twopcf:
        if key[:2] not in prefixes:
            continue
        block = prefixes[key[:2]]
        bounds = overrides.get(key, cfg.scale_cuts[block])
        if bounds is None:
            raise ValueError(f"'scale_cuts' gives no theta range for {'-'.join(map(str, key))}")
        scale_cuts[key] = _theta_range(bounds, block, "-".join(map(str, key)))
    return scale_cuts


def read_correlation_functions(path: str) -> tuple[np.ndarray, dict[tuple, np.ndarray]]:
    """Separations in arcmin and correlation functions of a LE3 2D TPCF file."""
    raw = el.le3.twopcf_wl.correlation_functions(path)
    if not raw:
        raise ValueError(f"No 2D correlation function found in '{path}'")
    theta = np.asarray(next(iter(raw.values())).theta, float)
    return theta, {tuple(key): np.asarray(value.array, float) for key, value in raw.items()}


def build_likelihood(cfg: RunConfig):
    """Build the real-space photometric likelihood instance for this run.

    Parameters
    ----------
    cfg: RunConfig
        The run's parsed configuration.

    Returns
    -------
    likelihood: EuclidLikelihood_WL, _GCph, _GGL, _2x2pt or _3x2pt
        Of `cloelike.EuclidLikelihood_photo_2pcf`.
    """
    kind = cfg.probe["kind"]
    data_cfg = cfg.data
    uses_pos = any("POS" in PHOTO_BLOCKS[block] for block in _BLOCKS[kind])
    uses_she = any("SHE" in PHOTO_BLOCKS[block] for block in _BLOCKS[kind])

    z_grid = np.linspace(
        data_cfg.z_grid["min"], data_cfg.z_grid["max"], int(data_cfg.z_grid["num"])
    )
    dndz_pos = _load_nz(data_cfg.nz_positions, z_grid)
    dndz_she = _load_nz(data_cfg.nz_shear, z_grid)
    n_pos, n_she = dndz_pos.shape[0], dndz_she.shape[0]

    if data_cfg.twopcf is not None:
        theta, twopcf = read_correlation_functions(data_cfg.twopcf)
        keys = _data_keys(kind, n_pos, n_she)
        twopcf = {key: value for key, value in twopcf.items() if key in keys}
    else:
        if data_cfg.theta is None:
            raise ValueError("'data' section needs 'twopcf', or 'theta' to build the data vector")
        theta = binning_theta(data_cfg.theta)
        twopcf = _template_twopcf(kind, n_pos, n_she, len(theta))
    theta = np.asarray(theta, float)
    scale_cuts = _build_scale_cuts(cfg, twopcf) if cfg.scale_cuts else None

    data: dict[str, Any] = {"2pcf": twopcf, "theta": theta, "z_arr": z_grid}
    if data_cfg.covariance is not None:
        covariance = _load_covariance(data_cfg.covariance, data_cfg.covariance_key)
        sizes = _block_sizes(n_pos, n_she, len(theta))
        n_data = sum(sizes[stat] for stat in _statistics(kind))
        if covariance.shape != (n_data, n_data):
            raise ValueError(
                f"Covariance of shape {covariance.shape} does not match the "
                f"{kind} data vector ({n_data})"
            )
        data["cov"] = covariance
    if uses_pos:
        data["dndz_pos"] = dndz_pos
    if uses_she:
        data["dndz_she"] = dndz_she

    ell_max = int(data_cfg.ell_max_integration or DEFAULT_ELL_MAX_INTEGRATION)
    background_name = cfg.probe["background"]
    engines = {
        "Background": load_background(background_name),
        "LinPerturbations": load_perturbations(
            cfg.probe.get("linear_perturbations", background_name)
        ),
        "NonLinPerturbations": load_nonlinear_perturbations(
            cfg.probe["nonlinear_perturbations"], cfg.probe.get("baryon_boost")
        ),
    }
    return _LIKELIHOODS[kind](
        data=data,
        settings=_build_settings(cfg, scale_cuts),
        **engines,
        ells_integration=np.arange(2, ell_max + 1),
        mode="decoupled",
    )


def split_vector(vector: np.ndarray, kind: str, n_pos: int, n_she: int, n_theta: int) -> dict[tuple, np.ndarray]:
    """Correlation functions keyed as the data, from the flat theory vector
    ``[xi_+ | xi_- | gamma_t | w]``."""
    vector = np.asarray(vector, float)
    sizes = _block_sizes(n_pos, n_she, n_theta)
    expected = sum(sizes[stat] for stat in _statistics(kind))
    if vector.size != expected:
        raise ValueError(f"Theory vector of length {vector.size} does not match {kind} ({expected})")
    out = _template_twopcf(kind, n_pos, n_she, n_theta)
    start = 0
    for block in _BLOCKS[kind]:
        pairs = _pairs(block, n_pos, n_she)
        for stat in _STATISTICS[block]:
            for i, j in pairs:
                key = (*PHOTO_BLOCKS[block], i, j)
                values = vector[start:start + n_theta]
                start += n_theta
                if stat == "xi_plus":
                    out[key][0, 0] = values
                elif stat == "xi_minus":
                    out[key][1, 1] = values
                elif stat == "gamma_t":
                    out[key][0] = values
                else:
                    out[key][:] = values
    return out


def theory_spectra(likelihood, cfg: RunConfig, parameters: dict[str, Any]) -> dict[str, Any]:
    """Theory correlation functions at ``parameters``, keyed as the data."""
    vector = likelihood.get_theory_vector_full(parameters)
    theta = np.asarray(likelihood.data["theta"], float)
    n_pos = likelihood.data["dndz_pos"].shape[0] if "dndz_pos" in likelihood.data else 0
    n_she = likelihood.data["dndz_she"].shape[0] if "dndz_she" in likelihood.data else 0
    return {
        "theta": theta,
        "twopcf": split_vector(vector, cfg.probe["kind"], n_pos, n_she, theta.size),
    }


def write_spectra(path: str, spectra: dict[str, Any]) -> None:
    """Write the output of `theory_spectra` as a LE3 2D TPCF file."""
    theta = np.asarray(spectra["theta"], float)
    weight = np.ones_like(theta)
    el.le3.twopcf_wl.correlation_functions.write(
        path,
        {
            key: el.le3.twopcf_wl.TwoPointCorrelationFunction(
                value, theta=theta, axis=(value.ndim - 1,), weight=weight
            )
            for key, value in spectra["twopcf"].items()
        },
    )
