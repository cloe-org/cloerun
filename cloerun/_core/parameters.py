"""Turn one flat sampler point into the full parameter dict needed by `loglike`."""

from __future__ import annotations

from typing import Any

import numpy as np

from cloerun._core.priors import ParameterLayout


def build_parameter_dict_for_loglike(
    flat_sample: dict[str, float],
    layout: ParameterLayout,
    fixed_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct the full parameter dictionary requested by `loglike` starting
    from a given flat sample.

    Parameters
    ----------
    flat_sample: dict[str, float]
        One flat point from the sampler.
    layout: ParameterLayout
        How to reassemble ``flat_sample``.
    fixed_parameters: dict[str, Any]
        Default values for the parameters that are not sampled.
    
    Returns
    -------
    parameters: dict[str, Any]
        Full parameter dictionary consistent with the layout expected by `loglike`.
    """
    parameters = dict(fixed_parameters)

    for name in layout.scalar_names:
        parameters[name] = flat_sample[name]

    for base_name, bin_entries in layout.vector_names.items():
        parameters[base_name] = np.array(
            [
                flat_sample[entry] if isinstance(entry, str) else entry
                for entry in bin_entries
            ]
        )

    return parameters
