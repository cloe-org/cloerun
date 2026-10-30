"""Wrapper for `cloelike.EuclidLikelihood_GCspectro_Pls`."""

from __future__ import annotations

from typing import Any

from euclidlib.le3.pk_gc import (
    power_spectrum_multipole_covariance,
    power_spectrum_multipole_mixing_matrix,
    power_spectrum_multipoles,
)

from cloelike.EuclidLikelihood_GCspectro_Pls import EuclidLikelihood_GCspectro_Pls

from cloerun._core.config import RunConfig
from cloerun._core.loader import load_background, load_perturbations, load_spectro_power
from cloerun._core.priors import extract_AM_priors

_BACKGROUND_PACKAGES = {
    "CAMB": "camb",
}
_PERTURBATIONS_PACKAGES = {
    "CAMB": "camb",
    "HMcode2020Emu": "HMcode2020Emu",
}
_SPECTRO_POWER_PACKAGES = {
    "COMET": "comet-emu",
    "PBJ": "pbjcosmo",
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
    return cfg.data.labels


def _load_raw_data(cfg: RunConfig) -> dict[str, Any]:
    """Read the raw `euclidlib` data vectors, covariances and (optional)
    mixing matrices for every redshift bin, in the nested-dict layout
    `EuclidLikelihood_GCspectro_Pls` expects.

    Parameters
    ----------
    cfg: RunConfig
        The run's parsed configuration.

    Returns
    -------
    data: dict[str, Any]
        Data dictionary in the form expected by the likelihood, i.e.
        `{"GCspectro": {label: {"datavec": ..., "covariance": ..., ["mixing": ...]}}}`.
    """
    data = cfg.data
    if data.datavec_FS_template is None:
        raise ValueError(
            "'data' section is missing required key 'datavec_FS' for "
            "probe 'GCspectro_Pls'"
        )

    datavec = power_spectrum_multipoles(data.datavec_FS_template, *data.labels)
    covariance = power_spectrum_multipole_covariance(data.covariance_template, *data.labels)
    mixing = (
        power_spectrum_multipole_mixing_matrix(data.mixing_template, *data.labels)
        if data.mixing_template is not None
        else None
    )

    result: dict[str, Any] = {"GCspectro": {}}
    for i, label in enumerate(data.labels):
        key = ("SPE", "SPE", i, i)
        entry = {"datavec": datavec[key], "covariance": covariance[key]}
        if mixing is not None:
            entry["mixing"] = mixing[key]
        result["GCspectro"][label] = entry
    return result


def _build_settings(cfg: RunConfig, h_fid: float) -> dict[str, Any]:
    """Convert scale cuts from h/Mpc to 1/Mpc, since the likelihood expects the latter.

    Parameters
    ----------
    cfg: RunConfig
        The run's parsed configuration.
    h_fid: float
        Fiducial dimensionless Hubble constant.

    Returns
    -------
    settings: dict[str, Any]
        Settings dictionary in the form expected by the likelihood, i.e.
        `{"GCspectro": {"scale_cuts": label: {ell: [min, max]}}}`.
    """
    scale_cuts = {
        label: {
            ell: [bound * h_fid for bound in bounds]
            for ell, bounds in ell_bounds.items()
        }
        for label, ell_bounds in cfg.scale_cuts.items()
    }
    return {"GCspectro": {"scale_cuts": scale_cuts}}


def _load_perturbations_if_needed(
    SpectroPower: type, perturbations_name: str, TCM_params: dict[str, Any]
):
    """Return the `Perturbations` class this run needs, or `None`.

    A `Perturbations` instance is needed either because `SpectroPower`'s
    theory itself requires one (PBJ), or because `TCM_params` needs one to
    compute sigma12 (even for COMET, which otherwise doesn't need one) --
    see `EuclidLikelihood_GCspectro_Pls`'s own `TCM_params` docstring.

    Parameters
    ----------
    SpectroPower: type
        The selected SpectroPower class.
    perturbations_name: str
        The config's ``probe.perturbations`` (or ``probe.background`` if
        not given explicitly).
    TCM_params: dict[str, Any]
        The config's ``probe.TCM_params`` section (possibly empty).

    Returns
    -------
    perturbations: type or None
    """
    if SpectroPower.NLcode != "PBJ" and not TCM_params:
        return None
    return load_perturbations(perturbations_name)


def build_likelihood(cfg: RunConfig) -> EuclidLikelihood_GCspectro_Pls:
    """Build the `EuclidLikelihood_GCspectro_Pls` instance for this run.

    Parameters
    ----------
    cfg: RunConfig
        The run's parsed configuration.

    Returns
    -------
    likelihood: EuclidLikelihood_GCspectro_Pls
        The built likelihood instance.
    """
    if not cfg.scale_cuts:
        raise ValueError(
            "'scale_cuts' is required for probe 'GCspectro_Pls' (one entry "
            "per data.redshifts)"
        )

    data = _load_raw_data(cfg)

    first_label = cfg.data.labels[0]
    fiducial_h = (
        data["GCspectro"][first_label]["datavec"].fiducial_cosmology["H0"] / 100.0
    )
    settings = _build_settings(cfg, fiducial_h)

    Background = load_background(cfg.probe["background"])
    spectro_power_cfg = cfg.probe["spectro_power"]
    SpectroPower = load_spectro_power(
        spectro_power_cfg["code"], spectro_power_cfg.get("rsd_model")
    )
    TCM_params = cfg.probe.get("TCM_params") or None
    perturbations_name = cfg.probe.get("perturbations", cfg.probe["background"])
    Perturbations = _load_perturbations_if_needed(
        SpectroPower, perturbations_name, TCM_params
    )

    AM_priors = extract_AM_priors(cfg.sampled_parameters, cfg.data.labels)

    likelihood = EuclidLikelihood_GCspectro_Pls(
        data=data,
        settings=settings,
        Background=Background,
        SpectroPower=SpectroPower,
        Perturbations=Perturbations,
        AM_priors=AM_priors,
        num_mocks=cfg.data.num_mocks,
        TCM_params=TCM_params,
    )
    # Not a constructor argument of the likelihood: read back by `loglike`,
    # which passes it to `loglike_AM` (Jeffreys priors on the analytically
    # marginalised parameters).
    likelihood.use_Jeffreys = bool(cfg.probe.get("use_Jeffreys", False))
    return likelihood


def loglike(likelihood: EuclidLikelihood_GCspectro_Pls, parameters: dict[str, Any]) -> float:
    """Evaluate log-likelihood on the given parameters.

    Parameters
    ----------
    likelihood: EuclidLikelihood_GCspectro_Pls
        An instance of the likelihood.
    parameters: dict[str, Any]
        Full parameter dictionary in the format returned by
        `cloerun._core.parameters.build_parameter_dict_for_loglike`.

    Returns
    -------
    loglike: float
        The log-likelihood value.
    """
    if likelihood.AM_priors:
        return likelihood.loglike_AM(
            parameters, use_Jeffreys=getattr(likelihood, "use_Jeffreys", False)
        )
    return likelihood.loglike(parameters)


def vector_parameter_names(likelihood: EuclidLikelihood_GCspectro_Pls) -> set[str]:
    """Return names of nuisance parameters (one value per observable bin)

    Parameters
    ----------
    likelihood: EuclidLikelihood_GCspectro_Pls
        A likelihood instance built by `build_likelihood`.

    Returns
    -------
    names: set[str]
        Names of nuisance parameters that are individual per observable bin.
    """
    return set(likelihood.RSD_parameter_names) | set(
        likelihood.noise_syst_parameter_names
    )


def tracked_packages(cfg: RunConfig) -> list[str]:
    """Physics-engine package(s) this run's `background`/`spectro_power` uses.

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
    spectro_power_cfg = cfg.probe["spectro_power"]
    code = spectro_power_cfg["code"]
    if code in _SPECTRO_POWER_PACKAGES:
        packages.append(_SPECTRO_POWER_PACKAGES[code])
    TCM_params = cfg.probe.get("TCM_params")
    perturbations_needed = code == "PBJ" or bool(TCM_params)
    if perturbations_needed:
        perturbations_name = cfg.probe.get("perturbations", background_name)
        package = _PERTURBATIONS_PACKAGES.get(perturbations_name)
        if package is not None and package not in packages:
            packages.append(package)
    return packages
