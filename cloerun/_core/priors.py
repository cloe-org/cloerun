"""Module for reading and building the prior distributions of sampled parameters"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import scipy.stats


@dataclass
class ParameterLayout:
    """Structure of sampled parameters, whether scalar, per-bin,
    or analytically marginalised.

    Attributes
    ----------
    scalar_names: list[str]
        Names of the scalar sampled parameters.
    vector_names: dict[str, list[str or float]]
        Dictionary mapping each per-bin parameter name to a list with one
        entry per bin: either the bin's flat component name (``str``, to be
        read from the flat sample), or its analytically-marginalised
        placeholder value (``float``, the ``dist: AM`` spec's ``loc``) for a
        bin the sampler never draws.
    """

    scalar_names: list[str] = field(default_factory=list)
    vector_names: dict[str, list[str | float]] = field(default_factory=dict)


@dataclass
class Priors:
    """Container for the prior distribution of sampled parameters and layout
    information.

    `distributions` maps each flat sampled parameter name to its frozen
    `scipy.stats` distribution, in the order parameters were declared.
    `layout` records how to reassemble per-bin ("vector") parameters.

    Attributes
    ----------
    distributions: dict[str, Any]
        Dictionary with the association of sampled parameters to the corresponding
        `scipy.stats` distribution.
    layout: ParameterLayout
        Layout information for reassembling per-bin parameters.
    AM_priors: dict[str, dict[str, list[float]]]
        Analytically-marginalised (``dist: AM``) entries of
        `sampled_parameters`, re-keyed by redshift label and ready to pass
        as `EuclidLikelihood_GCspectro_Pls`'s ``AM_priors`` constructor
        argument -- see `extract_AM_priors`. Empty if none are configured.
    """

    distributions: dict[str, Any]
    layout: ParameterLayout
    AM_priors: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        """Flat parameter names."""
        return list(self.distributions.keys())


def _is_AM_spec(spec: Any) -> bool:
    """Whether one per-bin spec requests analytical marginalisation.
    
    Parameters
    ----------
    spec: Any
        A parameter specification, as indicated in the config yaml file.

    Returns
    -------
    is_AM: bool
        True if the spec is a dict with ``dist: AM``, False otherwise.
    """
    return isinstance(spec, dict) and spec.get("dist") == "AM"


def _distribution(spec: dict[str, Any]):
    """Turn one parameter spec into a frozen `scipy.stats` distribution.

    Parameters
    ----------
    spec: dict[str, Any]
        One parameter spec, which can be either:
        flat,  e.g. ``{"dist": "flat", "min": 0, "max": 1}``
        gaussian, e.g. ``{"dist": "norm", "loc": 0, "scale": 1}``.

    Returns
    -------
    dist: scipy.stats distribution
        Frozen distribution object.
    """
    spec = dict(spec) # Shallow copy to avoid modifying input spec (needed when broadcasting prior)
    dist_name = spec.pop("dist")
    if dist_name == "flat":
        lo = spec.pop("min")
        hi = spec.pop("max")
        return scipy.stats.uniform(loc=lo, scale=hi - lo)
    return getattr(scipy.stats, dist_name)(**spec)


def extract_AM_priors(
    sampled_parameters: dict[str, Any], labels: list[str]
) -> dict[str, dict[str, list[float]]]:
    """Extract the analytically-marginalised (``dist: AM``) entries of
    `sampled_parameters`, re-keyed by redshift labels.

    Parameters
    ----------
    sampled_parameters: dict[str, Any]
        The raw section of the sampled parameters in the config yaml file.
    labels: list[str]
        Redshift bin labels, in data order -- a list entry at index ``i``
        refers to ``labels[i]``.

    Returns
    -------
    AM_priors: dict[str, dict[str, list[float]]]
        ``{label: {name: [mean, std]}}``; empty if none are configured.
    """
    AM_priors: dict[str, dict[str, list[float]]] = {}
    for name, spec in sampled_parameters.items():
        if isinstance(spec, list):
            bin_specs = spec
        elif _is_AM_spec(spec):
            bin_specs = [spec] * len(labels)
        else:
            continue
        for i, bin_spec in enumerate(bin_specs):
            if _is_AM_spec(bin_spec):
                AM_priors.setdefault(labels[i], {})[name] = [
                    bin_spec["loc"], bin_spec["scale"]
                ]
    return AM_priors


def build_priors(
    sampled_parameters: dict[str, Any],
    labels: list[str],
    vector_parameter_names: Iterable[str] = (),
) -> Priors:
    """Build a sampler-agnostic set of priors.

    Parameters
    ----------
    sampled_parameters: dict[str, Any]
        The raw section of the sampled parameters in the config yaml file.
    labels: list[str]
        Redshift bin labels, in data order -- also gives the number of bins
        used to broadcast a shared spec across `vector_parameter_names`, if
        applicable.
    vector_parameter_names: Iterable[str]
        Names of the nuisance parameters of the specific probe.

    Returns
    -------
    priors: Priors
        Frozen distributions and layout information for the sampled parameters.
    """
    vector_parameter_names = set(vector_parameter_names)
    n_bins = len(labels)
    AM_priors = extract_AM_priors(sampled_parameters, labels)
    distributions: dict[str, Any] = {}
    layout = ParameterLayout()

    for name, spec in sampled_parameters.items():
        if isinstance(spec, list):
            bin_specs = spec
        elif _is_AM_spec(spec) or name in vector_parameter_names:
            bin_specs = [spec] * n_bins
        else:
            distributions[name] = _distribution(spec)
            layout.scalar_names.append(name)
            continue

        bin_entries: list[str | float] = []
        for i, bin_spec in enumerate(bin_specs):
            if _is_AM_spec(bin_spec):
                bin_entries.append(bin_spec["loc"])
            else:
                bin_name = f"{name}[{i}]"
                distributions[bin_name] = _distribution(bin_spec)
                bin_entries.append(bin_name)
        layout.vector_names[name] = bin_entries

    return Priors(distributions=distributions, layout=layout, AM_priors=AM_priors)
