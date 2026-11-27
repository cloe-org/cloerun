"""Wrapper for the cross-correlations of Euclid photometric probes with CMB lensing
of `cloelike`."""

from __future__ import annotations

from typing import Any

import numpy as np
import euclidlib as el

from cloelike.EuclidLikelihood_CMBLensingCross import (
    EuclidLikelihood_CMBLxGC,
    EuclidLikelihood_CMBLxWL,
    EuclidLikelihood_CMBLx,
    EuclidLikelihood_5x2pt,
)

from cloerun._core.config import RunConfig, CMBL_BLOCKS
from cloerun._core.loader import (
    load_background,
    load_nonlinear_perturbations,
    load_perturbations,
)

from cloerun._core.probes.photo_cls import (
    _build_settings, _load_covariance, _load_nz, binning_ells,
    derived_outputs, loglike, obs_labels, parameter_names,
    set_data_vector_masked, theory_spectra, tracked_packages,
    use_noise_free_mock, vector_parameter_names, write_spectra,
)

_LIKELIHOODS = {
    "CMBLxWL": EuclidLikelihood_CMBLxWL,
    "CMBLxGC": EuclidLikelihood_CMBLxGC,
    "CMBLx": EuclidLikelihood_CMBLx,
    "5x2pt": EuclidLikelihood_5x2pt,
}

_BLOCKS = {
    "CMBLxWL": ("CMBLxWL",),
    "CMBLxGC": ("CMBLxGC",),
    "CMBLx": ("CMBLxWL", "CMBLxGC"),
    "5x2pt": ("CMBLxWL", "CMBLxGC", "WL", "GGL", "GCph"),
}

def _data_keys(kind: str, n_pos: int, n_she: int) -> list[tuple]:
    """Keys of every spectrum of the probe, as `cloelike` reads them.

    Parameters
    ----------
    kind: str
        Probe kind.
    n_pos: int
        Number of positions bins.
    n_she: int
        Number of shear bins.

    Returns
    -------
    keys: list[tuple]
    """
    pairs = {
        "WL": [(i, j) for i in range(1, n_she + 1) for j in range(i, n_she + 1)],
        "GGL": [(i, j) for i in range(1, n_pos + 1) for j in range(1, n_she + 1)],
        "GCph": [(i, j) for i in range(1, n_pos + 1) for j in range(i, n_pos + 1)],
        "CMBLxWL": [(1, i) for i in range(1, n_she + 1)],
        "CMBLxGC": [(1, i) for i in range(1, n_pos + 1)],
    }
    return [(*CMBL_BLOCKS[block], i, j) for block in _BLOCKS[kind] for i, j in pairs[block]]

def _build_scale_cuts(cfg: RunConfig, cells: dict) -> dict[tuple, list[float]]:
    """Multipole range of every spectrum of the probe, keyed as the data.

    Same as `photo_cls._build_scale_cuts`, with the CMB lensing blocks.

    Parameters
    ----------
    cfg: RunConfig
        The run's parsed configuration.
    cells: dict
        The angular power spectra, keyed as ``("SHE", "SHE", 1, 2)``.

    Returns
    -------
    scale_cuts: dict[tuple, list[float]]
    """
    kind = cfg.probe["kind"]
    missing = [block for block in _BLOCKS[kind] if block not in cfg.scale_cuts]
    if missing:
        raise ValueError(f"'scale_cuts' is missing the range of block(s) {missing}")
    prefixes = {CMBL_BLOCKS[block]: block for block in _BLOCKS[kind]}
    overrides = cfg.scale_cuts["overrides"]
    unknown = [key for key in overrides if key not in cells]
    if unknown:
        raise ValueError(f"'scale_cuts.overrides' has pairs not in the data: {unknown}")
    scale_cuts = {
        key: overrides.get(key, cfg.scale_cuts[prefixes[key[:2]]])
        for key in cells
        if key[:2] in prefixes
    }
    uncut = ["-".join(map(str, key)) for key, bounds in scale_cuts.items() if bounds is None]
    if uncut:
        raise ValueError(f"'scale_cuts' gives no multipole range for {uncut}")
    return scale_cuts

def _template_cells(kind: str, n_pos: int, n_she: int, n_ell: int) -> dict[tuple, np.ndarray]:
    """Zero spectra with the keys and shapes `cloelike` reads, standing for
    the data when the data vector is the theory at the fiducial.

    Parameters
    ----------
    kind: str
        Probe kind.
    n_pos: int
        Number of positions bins.
    n_she: int
        Number of shear bins.
    n_ell: int
        Number of multipoles per spectrum.

    Returns
    -------
    cells: dict[tuple, np.ndarray]
    """

    shapes = {("SHE", "SHE"): (1, 1, n_ell), 
              ("POS", "SHE"): (1, n_ell), 
              ("POS", "POS"): (n_ell,),
              ("CMBL", "SHE"): (1, n_ell), 
              ("CMBL", "POS"): (n_ell,),}
    return {key: np.zeros(shapes[key[:2]]) for key in _data_keys(kind, n_pos, n_she)}

def _block_sizes(n_pos: int, n_she: int, n_ell: int) -> dict[str, int]:
    """Length of each block of the data vector.

    Parameters
    ----------
    n_pos: int
        Number of positions bins.
    n_she: int
        Number of shear bins.
    n_ell: int
        Number of multipoles per spectrum.

    Returns
    -------
    sizes: dict[str, int]
    """
    return {
        "WL": n_she * (n_she + 1) // 2 * n_ell,
        "GGL": n_pos * n_she * n_ell,
        "GCph": n_pos * (n_pos + 1) // 2 * n_ell,
        "CMBLxWL": n_she * n_ell,
        "CMBLxGC": n_pos * n_ell,
    }
    
def _select_covariance(
    covariance: np.ndarray, kind: str, sizes: dict[str, int]
) -> np.ndarray:
    """Return the covariance of the probe's blocks.

    Parameters
    ----------
    covariance: np.ndarray
        Covariance of either exactly the probe's blocks, or of the full 5x2pt
        data vector (blocks ordered WL, GGL, GCph, CMBLxWL, CMBLxGC).
    kind: str
        Probe kind.
    sizes: dict[str, int]
        Length of each block, see `_block_sizes`.

    Returns
    -------
    covariance: np.ndarray
    """
    n_probe = sum(sizes[block] for block in _BLOCKS[kind])
    n_full = sum(sizes[block] for block in _BLOCKS["5x2pt"])
    if covariance.shape == (n_probe, n_probe):
        return covariance
    if covariance.shape != (n_full, n_full):
        raise ValueError(
            f"Covariance of shape {covariance.shape} matches neither the "
            f"{kind} data vector ({n_probe}) nor the 5x2pt one ({n_full})"
        )
    start = 0
    indices = []
    for block in _BLOCKS["5x2pt"]:
        if block in _BLOCKS[kind]:
            indices.append(np.arange(start, start + sizes[block]))
        start += sizes[block]
    indices = np.concatenate(indices)
    return covariance[np.ix_(indices, indices)]

def build_likelihood(cfg: RunConfig):
    """Build the photometric likelihood instance for this run.

    Parameters
    ----------
    cfg: RunConfig
        The run's parsed configuration.

    Returns
    -------
    likelihood: EuclidLikelihood_CMBLxGC, EuclidLikelihood_CMBLxWL, 
    EuclidLikelihood_CMBLx, EuclidLikelihood_5x2pt,
        The built likelihood instance.
    """
    kind = cfg.probe["kind"]
    data_cfg = cfg.data
    mode = cfg.probe.get("mode", "decoupled")
    uses_pos = any("POS" in CMBL_BLOCKS[block] for block in _BLOCKS[kind])
    uses_she = any("SHE" in CMBL_BLOCKS[block] for block in _BLOCKS[kind])

    z_grid = np.linspace(
        data_cfg.z_grid["min"], data_cfg.z_grid["max"], int(data_cfg.z_grid["num"])
    )
    dndz_pos = _load_nz(data_cfg.nz_positions, z_grid)
    dndz_she = _load_nz(data_cfg.nz_shear, z_grid)

    if data_cfg.cells is not None:
        cells = el.le3.pk_wl.angular_power_spectra(data_cfg.cells)
        prefixes = {CMBL_BLOCKS[block] for block in _BLOCKS[kind]}
        ells = np.asarray(next(cells[key] for key in cells if key[:2] in prefixes).ell)
    else:
        ells = binning_ells(data_cfg.ells)
        cells = _template_cells(kind, dndz_pos.shape[0], dndz_she.shape[0], len(ells))
    # The sdv mode gives no scale cuts nor covariance: cloelike only reads
    # them in the loglike, not in the theory.
    scale_cuts = _build_scale_cuts(cfg, cells) if cfg.scale_cuts else None

    data: dict[str, Any] = {"cells": cells, "ells": ells, "z_arr": z_grid}
    if data_cfg.covariance is not None:
        sizes = _block_sizes(dndz_pos.shape[0], dndz_she.shape[0], len(ells))
        data["cov"] = _select_covariance(
            _load_covariance(data_cfg.covariance, data_cfg.covariance_key), kind, sizes
        )
    if uses_pos:
        data["dndz_pos"] = dndz_pos
    if uses_she:
        data["dndz_she"] = dndz_she
    if mode == "coupled":
        if data_cfg.mixing is None:
            raise ValueError("'data.mixing' is required for 'probe.mode: coupled'")
        data["mixmat"] = el.le3.pk_wl.mixing_matrices(data_cfg.mixing)

    background_name = cfg.probe["background"]
    return _LIKELIHOODS[kind](
        data=data,
        settings=_build_settings(cfg, scale_cuts),
        Background=load_background(background_name),
        LinPerturbations=load_perturbations(
            cfg.probe.get("linear_perturbations", background_name)
        ),
        NonLinPerturbations=load_nonlinear_perturbations(
            cfg.probe["nonlinear_perturbations"], cfg.probe.get("baryon_boost")
        ),
        mode=mode,
    )