"""Wrapper for `cloelike.EuclidLikelihood_GCspectro_Pls_BAO`.
"""

from __future__ import annotations

from typing import Any

from euclidlib.le3.bao_gc import BAO_alphas
from euclidlib.le3.pk_gc import (
    power_spectrum_multipole_covariance,
    power_spectrum_multipole_mixing_matrix,
    power_spectrum_multipoles,
)

from cloelike.EuclidLikelihood_GCspectro_Pls_BAO import EuclidLikelihood_GCspectro_Pls_BAO

from cloerun._core.config import RunConfig
from cloerun._core.loader import load_background, load_spectro_power
from cloerun._core.priors import extract_AM_priors
from cloerun._core.probes.gcspectro_pls import (
    _build_settings,
    _load_perturbations_if_needed,
    loglike,
    tracked_packages,
    vector_parameter_names,
)

__all__ = [
    "obs_labels",
    "build_likelihood",
    "loglike",
    "vector_parameter_names",
    "tracked_packages",
]


def obs_labels(cfg: RunConfig) -> list[str]:
    """Observable bin labels in the order the data are read.

    Parameters
    ----------
    cfg: RunConfig
        The run's parsed configuration.

    Returns
    -------
    labels: list[str]
        The observable bin labels.
    """
    return cfg.data.labels


def _load_raw_data(cfg: RunConfig) -> dict[str, Any]:
    """Read the raw `euclidlib` full-shape data vectors, BAO alphas, joint
    covariances and (optional) mixing matrices for every redshift bin, in
    the nested-dict layout `EuclidLikelihood_GCspectro_Pls_BAO` expects.

    Parameters
    ----------
    cfg: RunConfig
        The run's parsed configuration.

    Returns
    -------
    data: dict[str, Any]
        Data dictionary in the form expected by the likelihood, i.e.
        ``{"GCspectro": {label: {"datavec_Pls": ..., "datavec_BAO": ...,
        "covariance": ..., ["mixing": ...]}}}``.
    """
    data = cfg.data
    if data.datavec_FS_template is None:
        raise ValueError(
            "'data' section is missing required key 'datavec_FS' for "
            "probe 'GCspectro_Pls_BAO'"
        )
    if data.datavec_BAO_template is None:
        raise ValueError(
            "'data' section is missing required key 'datavec_BAO' for "
            "probe 'GCspectro_Pls_BAO'"
        )

    datavec_pls = power_spectrum_multipoles(data.datavec_FS_template, *data.labels)
    datavec_bao = BAO_alphas(data.datavec_BAO_template, *data.bao_labels)
    covariance = power_spectrum_multipole_covariance(
        data.covariance_template, *data.labels, include_BAO=True
    )
    mixing = (
        power_spectrum_multipole_mixing_matrix(data.mixing_template, *data.labels)
        if data.mixing_template is not None
        else None
    )

    result: dict[str, Any] = {"GCspectro": {}}
    for i, label in enumerate(data.labels):
        key = ("SPE", "SPE", i, i)
        entry = {
            "datavec_Pls": datavec_pls[key],
            "datavec_BAO": datavec_bao[key],
            "covariance": covariance[key],
        }
        if mixing is not None:
            entry["mixing"] = mixing[key]
        result["GCspectro"][label] = entry
    return result


def build_likelihood(cfg: RunConfig) -> EuclidLikelihood_GCspectro_Pls_BAO:
    """Build the `EuclidLikelihood_GCspectro_Pls_BAO` instance for this run.

    Parameters
    ----------
    cfg: RunConfig
        The run's parsed configuration.

    Returns
    -------
    likelihood: EuclidLikelihood_GCspectro_Pls_BAO
        The built likelihood instance.
    """
    if not cfg.scale_cuts:
        raise ValueError(
            "'scale_cuts' is required for probe 'GCspectro_Pls_BAO' (one "
            "entry per data.redshifts)"
        )

    data = _load_raw_data(cfg)

    first_label = cfg.data.labels[0]
    fiducial_h = (
        data["GCspectro"][first_label]["datavec_Pls"].fiducial_cosmology["H0"] / 100.0
    )
    settings = _build_settings(cfg, fiducial_h)

    Background = load_background(cfg.probe["background"])
    spectro_power_cfg = cfg.probe["spectro_power"]
    SpectroPower = load_spectro_power(
        spectro_power_cfg["code"], spectro_power_cfg.get("rsd_model")
    )
    # tilde_params isn't supported yet by EuclidLikelihood_GCspectro_Pls_BAO,
    # so a Perturbations instance is only ever needed here for PBJ's own theory.
    Perturbations = _load_perturbations_if_needed(
        SpectroPower, cfg.probe["background"], None
    )

    AM_priors = extract_AM_priors(cfg.sampled_parameters, cfg.data.labels)

    return EuclidLikelihood_GCspectro_Pls_BAO(
        data=data,
        settings=settings,
        Background=Background,
        SpectroPower=SpectroPower,
        Perturbations=Perturbations,
        AM_priors=AM_priors,
        num_mocks=cfg.data.num_mocks,
    )
