"""A probe with a Gaussian likelihood, to test the runner without a theory
code. Registered in `cloerun._core.loader.PROBES` by the tests."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

SIGMA = 0.1


class ToyLikelihood:
    def __init__(self, cfg):
        self.data_vector = np.array([cfg.data.num_mocks or 0.0, 0.0])

    def theory(self, parameters):
        return np.array([parameters["a"], parameters["b_1"] + parameters["b_2"]])


def obs_labels(cfg):
    return []


def build_likelihood(cfg):
    return ToyLikelihood(cfg)


def vector_parameter_names(likelihood):
    return set()


def parameter_names(likelihood):
    return ["a", "b_1", "b_2", "c"]


def loglike(likelihood, parameters: dict[str, Any]) -> float:
    diff = likelihood.theory(parameters) - likelihood.data_vector
    return float(-0.5 * np.sum(diff**2) / SIGMA**2)


def use_noise_free_mock(likelihood, fiducial):
    vector = likelihood.theory(fiducial)
    set_data_vector_masked(likelihood, vector)
    return vector


def set_data_vector_masked(likelihood, vector):
    likelihood.data_vector = vector


def theory_spectra(likelihood, cfg, parameters):
    return {"theory": likelihood.theory(parameters).tolist()}


def write_spectra(path, spectra):
    with open(path, "w") as f:
        json.dump(spectra, f)


def derived_outputs(likelihood, parameters, names):
    return [parameters["a"] + parameters["b_1"] for _ in names]


def tracked_packages(cfg):
    return []
