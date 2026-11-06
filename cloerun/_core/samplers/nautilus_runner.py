"""Runs the nautilus sampler."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from nautilus import Prior, Sampler

from cloerun._core.priors import Priors

#: Package(s) whose version matters when this sampler is chosen.
TRACKED_PACKAGES = ("nautilus-sampler",)

#: Log-likelihood of an MPI worker, set once per worker by `_init_worker`.
_LOGLIKE = None


def _init_worker(loglike_fn):
    """Install the log-likelihood in an MPI worker, so that it is unpickled
    (and the likelihood rebuilt) once per worker, not once per evaluation."""
    global _LOGLIKE
    _LOGLIKE = loglike_fn


def _worker_loglike(*args):
    """Evaluate the log-likelihood installed by `_init_worker`."""
    return _LOGLIKE(*args)


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

    With `pool: mpi` the likelihood is evaluated in the workers of an
    `mpi4py.futures.MPIPoolExecutor`; launch with
    ``mpirun -n N python -m mpi4py.futures -m cloerun chain config.yaml``
    (N - 1 workers).
    """
    filepath = sampler_settings.get("filepath")
    if filepath is not None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    executor = None
    if sampler_settings.get("pool") == "mpi":
        from mpi4py.futures import MPIPoolExecutor

        executor = MPIPoolExecutor(initializer=_init_worker, initargs=(loglike_fn,))
        sampler_settings = {**sampler_settings, "pool": executor}
        loglike_fn = _worker_loglike
    prior = _build_nautilus_prior(priors)
    sampler = Sampler(prior, loglike_fn, **sampler_settings)
    try:
        sampler.run(**{"verbose": True, **(run_settings or {})})
    finally:
        if executor is not None:
            executor.shutdown()
    if n_derived == 0:
        points, logweight, loglikelihood = sampler.posterior()
        return points, logweight, loglikelihood, None
    points, logweight, loglikelihood, blobs = sampler.posterior(return_blobs=True)
    if blobs.dtype.names is not None:
        derived = np.column_stack([blobs[name] for name in blobs.dtype.names])
    else:
        derived = np.asarray(blobs, dtype=float).reshape(len(points), n_derived)
    return points, logweight, loglikelihood, derived
