"""Derived-parameter relations that mutate the parameter dictionary in place.
Applied after the flat sample is reassembled into the full parameter dictionary
and before the likelihood is evaluated."""

from __future__ import annotations

import numpy as np


def bG2_coevolution(parameters: dict) -> None:
    r"""Coevolution bias relation for the tidal bias, fixing
    :math:`b_{G_2} = -\frac{2}{7}(b_1 - 1)`.

    Parameters
    ----------
    parameters: dict
        Parameter dictionary containing the key ``b1``; ``bG2`` is
        overwritten (or added) in place with the derived value.
    """
    parameters["bG2"] = -2.0 / 7.0 * (parameters["b1"] - 1.0)


def bG2_excursion_set(parameters: dict) -> None:
    r"""Excursion-set-derived relation for the tidal bias, fixing
    :math:`b_{G_2} = 0.524 - 0.547\, b_1 + 0.046\, b_1^2`.

    Parameters
    ----------
    parameters: dict
        Parameter dictionary containing the key ``b1``; ``bG2`` is
        overwritten (or added) in place with the derived value.
    """
    b1 = parameters["b1"]
    parameters["bG2"] = 0.524 - 0.547 * b1 + 0.046 * b1**2


def bGam3_coevolution(parameters: dict) -> None:
    r"""Coevolution bias relation for the third-order tidal bias, fixing
    :math:`b_{\Gamma_3} = -\frac{1}{6}(b_1 - 1) - \frac{5}{2} b_{G_2}`.

    WARNING: if ``bG2`` is itself derived, this relation must be listed *after*
    the deriving relation under `derived_parameters` in the config yaml.

    Parameters
    ----------
    parameters: dict
        Parameter dictionary containing the keys ``b1`` and ``bG2``;
        ``bGam3`` is overwritten (or added) in place with the derived value.
    """
    parameters["bGam3"] = (
        -1.0 / 6.0 * (parameters["b1"] - 1.0) - 5.0 / 2.0 * parameters["bG2"]
    )


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
