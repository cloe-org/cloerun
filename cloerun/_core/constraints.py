"""Prior constraints: `(parameters: dict) -> bool` functions
rejecting a sampled point outright, before the likelihood is evaluated."""

from __future__ import annotations


def w0_wa_negative(parameters: dict) -> bool:
    """Require ``w0 + wa <= 0``, i.e. dark energy behaves as a repulsive force
    at high redshift.

    Parameters
    ----------
    parameters: dict
        Parameter dictionary containing the keys ``w0`` and ``wa``.

    Returns
    -------
    condition: bool
        ``True`` if the sampled point satisfies the constraint, ``False``
        otherwise.
    """
    return parameters["w0"] + parameters["wa"] <= 0
