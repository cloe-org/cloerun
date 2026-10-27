"""Sampler backends. Only `nautilus` is implemented today.

Each sampler module exposes:

- `run(loglike_fn, priors, sampler_settings, run_settings=None, n_derived=0)`,
  returning the raw `(points, logweight, loglikelihood, derived)` arrays
  (`derived` is `None` without derived outputs) -- turning
  those into a `cloerun._core.chain_data.Chain` and saving it happens centrally in
  `runner.py`, not here, so the on-disk chain format never depends on which
  sampler produced it.
- `TRACKED_PACKAGES`, the package(s) whose version matters when this sampler
  is chosen (e.g. `("nautilus-sampler",)`) -- see `cloerun._core.versions`.

See `cloerun._core.loader.SAMPLERS` for how a run is dispatched to one by name
from the config's `sampler.kind` key.
"""
