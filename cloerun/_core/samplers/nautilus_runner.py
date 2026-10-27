"""Runs the nautilus sampler."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from nautilus import Prior, Sampler

from cloerun._core.priors import Priors

#: Package(s) whose version matters when this sampler is chosen.
TRACKED_PACKAGES = ("nautilus-sampler",)


def _build_nautilus_prior(priors: Priors) -> Prior:
    """Translate a sampler-agnostic `Priors` into a `nautilus.Prior`."""
    prior = Prior()
    for name, dist in priors.distributions.items():
        prior.add_parameter(name, dist=dist)
    return prior


def run(
    loglike_fn: Callable[[dict[str, float]], Any],
    priors: Priors,
    sampler_settings: dict[str, Any],
    run_settings: dict[str, Any] | None = None,
    n_derived: int = 0,
):
    """Run nautilus and return the raw `(points, logweight, loglikelihood,
    derived)` arrays.

    Turning these into a `cloerun._core.chain_data.Chain` (and saving it) is not this
    module's job -- that happens once, centrally, in `runner.py`, so the
    on-disk chain format never depends on which sampler produced it.

    `sampler_settings` is passed straight through as `Sampler.__init__` kwargs
    (e.g. `n_live`, `filepath`, `pool`, `resume`); `run_settings` (from the
    config's `sampler.run_settings`, e.g. `f_live`, `n_eff`, `timeout`,
    `verbose`) is passed to `sampler.run()`. With `n_derived > 0`,
    `loglike_fn` returns the log-likelihood followed by `n_derived` derived
    quantities, stored by nautilus as blobs and returned as `derived`, of
    shape `(n_samples, n_derived)` (`None` otherwise).
    """
    filepath = sampler_settings.get("filepath")
    if filepath is not None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    prior = _build_nautilus_prior(priors)
    sampler = Sampler(prior, loglike_fn, **sampler_settings)
    sampler.run(**{"verbose": True, **(run_settings or {})})
    if n_derived == 0:
        points, logweight, loglikelihood = sampler.posterior()
        return points, logweight, loglikelihood, None
    points, logweight, loglikelihood, blobs = sampler.posterior(return_blobs=True)
    if blobs.dtype.names is not None:
        derived = np.column_stack([blobs[name] for name in blobs.dtype.names])
    else:
        derived = np.asarray(blobs, dtype=float).reshape(len(points), n_derived)
    return points, logweight, loglikelihood, derived
