"""Wrapper for `cloelike.EuclidLikelihood_GCspectro_BAO`."""

from __future__ import annotations

from typing import Any

from euclidlib.le3.bao_gc import BAO_alphas, BAO_alphas_covariance

from cloelike.EuclidLikelihood_GCspectro_BAO import EuclidLikelihood_GCspectro_BAO

from cloerun._core.config import RunConfig
from cloerun._core.loader import load_background, load_perturbations

_BACKGROUND_PACKAGES = {
    "CAMB": "camb",
}


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
    return cfg.data.bao_labels


def _load_raw_data(cfg: RunConfig) -> dict[str, Any]:
    """Read the raw `euclidlib` BAO alphas and covariances for every
    redshift bin, in the nested-dict layout `EuclidLikelihood_GCspectro_BAO`
    expects.

    Parameters
    ----------
    cfg: RunConfig
        The run's parsed configuration.

    Returns
    -------
    data: dict[str, Any]
        Data dictionary in the form expected by the likelihood, i.e.
        `{"GCspectro": {label: {"datavec": ..., "covariance": ...}}}`.
    """
    data = cfg.data
    if data.datavec_BAO_template is None:
        raise ValueError(
            "'data' section is missing required key 'datavec_BAO' for "
            "probe 'GCspectro_BAO'"
        )

    datavec = BAO_alphas(data.datavec_BAO_template, *data.bao_labels)
    covariance = BAO_alphas_covariance(data.covariance_template, *data.bao_labels)

    result: dict[str, Any] = {"GCspectro": {}}
    for i, label in enumerate(data.bao_labels):
        key = ("SPE", "SPE", i, i)
        result["GCspectro"][label] = {
            "datavec": datavec[key],
            "covariance": covariance[key],
        }
    return result


def build_likelihood(cfg: RunConfig) -> EuclidLikelihood_GCspectro_BAO:
    """Build the `EuclidLikelihood_GCspectro_BAO` instance for this run.

    Parameters
    ----------
    cfg: RunConfig
        The run's parsed configuration.

    Returns
    -------
    likelihood: EuclidLikelihood_GCspectro_BAO
        The built likelihood instance.
    """
    data = _load_raw_data(cfg)
    Background = load_background(cfg.probe["background"])
    LinearPerturbations = load_perturbations(cfg.probe["background"])

    return EuclidLikelihood_GCspectro_BAO(
        data=data,
        Background=Background,
        LinearPerturbations=LinearPerturbations,
        num_mocks=cfg.data.num_mocks,
    )


def loglike(likelihood: EuclidLikelihood_GCspectro_BAO, parameters: dict[str, Any]) -> float:
    """Evaluate log-likelihood on the given parameters.

    Parameters
    ----------
    likelihood: EuclidLikelihood_GCspectro_BAO
        An instance of the likelihood.
    parameters: dict[str, Any]
        Full parameter dictionary in the format returned by
        `cloerun._core.parameters.build_parameter_dict_for_loglike`.

    Returns
    -------
    loglike: float
        The log-likelihood value.
    """
    return likelihood.loglike(parameters)


def vector_parameter_names(likelihood: EuclidLikelihood_GCspectro_BAO) -> set[str]:
    """Return names of nuisance parameters (one value per observable bin)

    Parameters
    ----------
    likelihood: EuclidLikelihood_GCspectro_BAO
        A likelihood instance built by `build_likelihood`.

    Returns
    -------
    names: set[str]
        Names of nuisance parameters that are individual per observable bin.
    """
    return set()


def tracked_packages(cfg: RunConfig) -> list[str]:
    """Physics-engine package(s) this run's `background` uses.

    Parameters
    ----------
    cfg: RunConfig
        The run's parsed configuration.

    Returns
    -------
    packages: list[str]
        List of package names to track versions for.
    """
    packages = []
    background_name = cfg.probe["background"]
    if background_name in _BACKGROUND_PACKAGES:
        packages.append(_BACKGROUND_PACKAGES[background_name])
    return packages
