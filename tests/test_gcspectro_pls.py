from types import SimpleNamespace

import pytest

pytest.importorskip("cloelike")

from cloerun._core.probes import gcspectro_pls  # noqa: E402
from cloerun._core.probes.gcspectro_pls import (  # noqa: E402
    _load_perturbations_if_needed,
    loglike,
    tracked_packages,
)

TCM_PARAMS = {"b1": [0.5, 1, 1], "c0": [1, 2, 2]}


class _COMET:
    NLcode = "COMET"


class _PBJ:
    NLcode = "PBJ"


class _Likelihood:
    """Stand-in for the cloelike likelihood: tells which loglike was called."""

    def __init__(self, AM_priors, use_Jeffreys=False):
        self.AM_priors = AM_priors
        self.use_Jeffreys = use_Jeffreys

    def loglike(self, parameters):
        return "loglike"

    def loglike_AM(self, parameters, use_Jeffreys=False):
        return ("loglike_AM", use_Jeffreys)


def _probe(**extra):
    """`probe` section of a COMET VDG run."""
    return {
        "kind": "GCspectro_Pls",
        "background": "CAMB",
        "spectro_power": {"code": "COMET", "rsd_model": "VDG"},
        **extra,
    }


@pytest.fixture
def loaded(monkeypatch):
    """Replaces `load_perturbations`; returns the perturbations requested."""
    requested = []

    def load_perturbations(name):
        requested.append(name)
        return f"Perturbations[{name}]"

    monkeypatch.setattr(gcspectro_pls, "load_perturbations", load_perturbations)
    return requested


@pytest.mark.parametrize("TCM_params", [None, {}])
def test_load_perturbations_if_needed_COMET_without_TCM(loaded, TCM_params):
    assert _load_perturbations_if_needed(_COMET, "CAMB", TCM_params) is None
    assert loaded == []


def test_load_perturbations_if_needed_COMET_with_TCM(loaded):
    # COMET needs no linear P(k), but the TCM needs sigma12.
    perturbations = _load_perturbations_if_needed(_COMET, "HMcode2020Emu", TCM_PARAMS)
    assert perturbations == "Perturbations[HMcode2020Emu]"
    assert loaded == ["HMcode2020Emu"]


@pytest.mark.parametrize("TCM_params", [None, TCM_PARAMS])
def test_load_perturbations_if_needed_PBJ(loaded, TCM_params):
    # PBJ needs the linear P(k), with or without the TCM.
    assert _load_perturbations_if_needed(_PBJ, "CAMB", TCM_params) == "Perturbations[CAMB]"
    assert loaded == ["CAMB"]


@pytest.mark.parametrize("use_Jeffreys", [False, True])
def test_loglike_AM(use_Jeffreys):
    likelihood = _Likelihood(AM_priors={"1.0": {"c0": [0.0, 1.0]}}, use_Jeffreys=use_Jeffreys)
    assert loglike(likelihood, {}) == ("loglike_AM", use_Jeffreys)


def test_loglike_without_AM():
    assert loglike(_Likelihood(AM_priors={}), {}) == "loglike"


@pytest.mark.parametrize(
    "probe, packages",
    [
        (_probe(), ["camb", "comet-emu"]),
        (_probe(perturbations="HMcode2020Emu"), ["camb", "comet-emu"]),
        (_probe(TCM_params=TCM_PARAMS), ["camb", "comet-emu"]),
        (
            _probe(perturbations="HMcode2020Emu", TCM_params=TCM_PARAMS),
            ["camb", "comet-emu", "HMcode2020Emu"],
        ),
        (_probe(spectro_power={"code": "PBJ", "rsd_model": "EFT"}), ["camb", "pbjcosmo"]),
    ],
    ids=["COMET", "COMET_unused_perturbations", "COMET_TCM_CAMB", "COMET_TCM_HMemu", "PBJ"],
)
def test_tracked_packages(probe, packages):
    assert tracked_packages(SimpleNamespace(probe=probe)) == packages
