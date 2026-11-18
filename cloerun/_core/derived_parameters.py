"""Derived-parameter relations that mutate the parameter dictionary in place.
Applied after the flat sample is reassembled into the full parameter dictionary
and before the likelihood is evaluated."""

from __future__ import annotations

import numpy as np


def H0_from_h(parameters: dict) -> None:
    r"""Relation between the Hubble parameter $H_0$ and the dimensionless Hubble parameter $h$.

    Parameters
    ----------
    parameters: dict
        Parameter dictionary containing the key ``h``; ``H0`` is
        overwritten (or added) in place with the derived value.
    """
    parameters["H0"] = 100.0 * parameters["h"]


def Omega_cdm0_from_omch2(parameters: dict) -> None:
    r"""Relation between the CDM fractional density parameter $\Omega_{\text{cdm},0}$
    and the physical CDM density parameter $\omega_{\text{cdm}}$.

    Parameters
    ----------
    parameters: dict
        Parameter dictionary containing ``omch2`` and either ``h`` or
        ``H0``; ``Omega_cdm0`` is overwritten (or added) in place with the
        derived value.
    """
    h = parameters["h"] if "h" in parameters else parameters["H0"] / 100.0
    parameters["Omega_cdm0"] = parameters["omch2"] / h**2


def Omega_b0_from_ombh2(parameters: dict) -> None:
    r"""Relation between the baryon fractional density parameter $\Omega_{\text{b},0}$
    and the physical baryon density parameter $\omega_{\text{b}}$.

    Parameters
    ----------
    parameters: dict
        Parameter dictionary containing ``ombh2`` and either ``h`` or
        ``H0``; ``Omega_b0`` is overwritten (or added) in place with the
        derived value.
    """
    h = parameters["h"] if "h" in parameters else parameters["H0"] / 100.0
    parameters["Omega_b0"] = parameters["ombh2"] / h**2


def As_from_logA(parameters: dict) -> None:
    r"""Relation between the amplitude parameter $A_s$ and its logarithm $\ln(10^{10} A_s)$.

    Parameters
    ----------
    parameters: dict
        Parameter dictionary containing the key ``logA``; ``As`` is
        overwritten (or added) in place with the derived value.
    """
    parameters["As"] = np.exp(parameters["logA"]) * 1e-10
