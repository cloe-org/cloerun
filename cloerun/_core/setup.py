"""Shared set-up of a run: likelihood, parameters, fiducial point and the
log-likelihood handed to the samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cloerun._core.config import RunConfig
from cloerun._core.parameters import build_parameter_dict_for_loglike
from cloerun._core.priors import ParameterLayout, Priors, build_priors
from cloerun._core.templates import expand_bin_templates, prune_unused
from cloerun._core.loader import (
    DERIVED_PARAMETERS,
    load_constraint,
    load_derived_parameter,
    load_probe,
)


class LogLikelihood:
    """Log-likelihood of a run, as a function of one flat sampler point.

    It can be pickled, so that a sampler can evaluate it in worker processes
    (e.g. nautilus' ``pool``, also on macOS where workers are spawned): the
    likelihood is not pickled but rebuilt once in each worker from the config.

    Parameters
    ----------
    config: RunConfig
        The run's configuration.
    fixed_parameters: dict[str, Any]
        Values of the parameters that are not sampled, ``{i}`` templates
        already expanded.
    layout: ParameterLayout
        How to reassemble a flat sample, see `cloerun._core.priors.Priors`.
    data_vector: np.ndarray or None
        Masked data vector replacing the one of the data file, e.g. the
        noise-free mock.
    likelihood: object or None
        Likelihood already built from ``config`` in this process, to reuse.
    """

    def __init__(
        self,
        config: RunConfig,
        fixed_parameters: dict[str, Any],
        layout: ParameterLayout,
        data_vector: np.ndarray | None = None,
        likelihood=None,
    ):
        self.config = config
        self.fixed_parameters = fixed_parameters
        self.layout = layout
        self.data_vector = data_vector
        self._likelihood = likelihood

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_likelihood": None}

    @property
    def probe(self):
        """The probe module of the run."""
        return load_probe(self.config.probe["kind"])

    @property
    def likelihood(self):
        """The likelihood instance, built on first use."""
        if self._likelihood is None:
            self._likelihood = self.probe.build_likelihood(self.config)
            if self.data_vector is not None:
                self.probe.set_data_vector_masked(self._likelihood, self.data_vector)
        return self._likelihood

    def parameters(self, flat_sample: dict[str, float]) -> dict[str, Any]:
        """Full parameter dictionary of one flat sampler point, derived
        parameters included.

        Parameters
        ----------
        flat_sample: dict[str, float]
            One flat point from the sampler.

        Returns
        -------
        parameters: dict[str, Any]
        """
        parameters = build_parameter_dict_for_loglike(
            flat_sample, self.layout, self.fixed_parameters
        )
        for name, enabled in self.config.derived_parameters.items():
            if enabled:
                load_derived_parameter(name)(parameters)
        return parameters

    def __call__(self, flat_sample: dict[str, float]):
        """Log-likelihood of one flat sampler point.

        Returns
        -------
        loglike: float or tuple
            The log-likelihood, followed by the ``derived_outputs`` if any.
        """
        derived_names = self.config.derived_outputs
        failed = (-np.inf, *[np.nan] * len(derived_names)) if derived_names else -np.inf
        parameters = self.parameters(flat_sample)
        for name, enabled in self.config.constraints.items():
            if enabled and not load_constraint(name)(parameters):
                return failed
        try:
            loglike = self.probe.loglike(self.likelihood, parameters)
        except (ValueError, RuntimeError):
            return failed
        if not derived_names:
            return loglike
        derived = self.probe.derived_outputs(self.likelihood, parameters, derived_names)
        return (loglike, *derived)


def _fiducial_point(
    config: RunConfig, fixed_parameters: dict[str, Any], fiducial_parameters: dict[str, Any]
) -> dict[str, Any]:
    """Full parameter dictionary at the fiducial point.

    Parameters
    ----------
    config: RunConfig
        The run's configuration.
    fixed_parameters: dict[str, Any]
        Expanded ``fixed_parameters``.
    fiducial_parameters: dict[str, Any]
        Expanded ``fiducial_parameters``, the values of the sampled ones.

    Returns
    -------
    parameters: dict[str, Any]
    """
    parameters = {**fixed_parameters, **fiducial_parameters}
    # A relation is applied only if its input is part of the point: a config
    # may give the fiducial in the native parameters (Omega_cdm0, As, ...).
    for name, enabled in config.derived_parameters.items():
        trigger = DERIVED_PARAMETERS[name].get("trigger")
        if enabled and (trigger is None or trigger in parameters):
            load_derived_parameter(name)(parameters)
    return parameters


@dataclass
class Setup:
    """Everything a run needs, built once from its config.

    Attributes
    ----------
    config: RunConfig
        The run's configuration.
    probe: module
        The probe module, see `cloerun._core.probes`.
    likelihood: object
        The likelihood instance.
    fixed_parameters: dict[str, Any]
        ``fixed_parameters``, unused ones dropped and templates expanded.
    sampled_parameters: dict[str, Any]
        ``sampled_parameters``, likewise.
    fiducial_parameters: dict[str, Any]
        ``fiducial_parameters``, likewise.
    """

    config: RunConfig
    probe: Any
    likelihood: Any
    fixed_parameters: dict[str, Any]
    sampled_parameters: dict[str, Any]
    fiducial_parameters: dict[str, Any]

    @property
    def fiducial(self) -> dict[str, Any]:
        """Full parameter dictionary at the fiducial point."""
        return _fiducial_point(self.config, self.fixed_parameters, self.fiducial_parameters)

    def loglikelihood(self) -> tuple[LogLikelihood, Priors]:
        """The log-likelihood of a flat sampler point, and the priors of
        the sampled parameters.

        With ``data.noise_free_mock: true`` the data vector of the
        likelihood is first replaced with the theory at the fiducial point.

        Returns
        -------
        loglike: LogLikelihood
        priors: Priors
        """
        noise_free_mock = getattr(self.config.data, "noise_free_mock", False)
        if getattr(self.config.data, "cells", "") is None and not noise_free_mock:
            raise ValueError(
                "No data vector: give 'data.cells', or set 'data.noise_free_mock: true'"
            )
        data_vector = None
        if noise_free_mock:
            data_vector = self.probe.use_noise_free_mock(self.likelihood, self.fiducial)
        priors = build_priors(
            self.sampled_parameters,
            self.probe.obs_labels(self.config),
            self.probe.vector_parameter_names(self.likelihood),
        )
        loglike = LogLikelihood(
            self.config, self.fixed_parameters, priors.layout,
            data_vector=data_vector, likelihood=self.likelihood,
        )
        return loglike, priors


def prepare(config: RunConfig) -> Setup:
    """Build the likelihood of a run and resolve its parameters.

    Parameters
    ----------
    config: RunConfig
        A pre-loaded running configuration.

    Returns
    -------
    setup: Setup
    """
    probe = load_probe(config.probe["kind"])
    likelihood = probe.build_likelihood(config)

    fixed_parameters = config.fixed_parameters
    sampled_parameters = config.sampled_parameters
    fiducial_parameters = config.fiducial_parameters
    # Probes with one scalar parameter per bin expand the '{i}' templates.
    if hasattr(probe, "parameter_names"):
        names = probe.parameter_names(likelihood)
        if config.prune_parameters:
            keep = [r["trigger"] for r in DERIVED_PARAMETERS.values() if "trigger" in r]
            fixed_parameters, dropped_fixed = prune_unused(fixed_parameters, names, keep)
            sampled_parameters, dropped_sampled = prune_unused(sampled_parameters, names, keep)
            fiducial_parameters, _ = prune_unused(fiducial_parameters, names, keep)
            if dropped_fixed or dropped_sampled:
                print(
                    "Parameters not read by the likelihood, dropped: "
                    f"{dropped_sampled + dropped_fixed}"
                )
        fixed_parameters = expand_bin_templates(fixed_parameters, names, "fixed_parameters")
        sampled_parameters = expand_bin_templates(
            sampled_parameters, names, "sampled_parameters"
        )
        fiducial_parameters = expand_bin_templates(
            fiducial_parameters, names, "fiducial_parameters"
        )
    return Setup(
        config, probe, likelihood, fixed_parameters, sampled_parameters, fiducial_parameters
    )
