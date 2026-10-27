"""Chains: sample the posterior of a run and save it, in the same format
whatever the sampler (see `Chain`).

    cloerun chain campaign.yaml --run chain_WL_baseline
    cloerun chain campaign.yaml --test     # one loglike at the fiducial point

A run writes ``<output.directory>/<output.name>.npz``, the chain, and
``<output.name>.config.yaml``, its config with the versions of the packages
it ran with.
"""

from __future__ import annotations

import argparse
import time

from cloerun._core.chain_data import Chain
from cloerun._core.config import RunConfig
from cloerun._core.loader import load_sampler
from cloerun._core.setup import prepare
from cloerun._core.versions import write_config_with_versions
from cloerun.campaign import add_run_arguments, load_runs

__all__ = ["Chain", "run_chain", "test_loglike"]


def run_chain(config: RunConfig) -> Chain:
    """Sample the posterior of a run.

    Parameters
    ----------
    config: RunConfig
        A pre-loaded running configuration.

    Returns
    -------
    chain: Chain
        The sampled chain, also saved to the output directory of the config.
    """
    setup = prepare(config)
    loglike, priors = setup.loglikelihood()
    sampler = load_sampler(config.sampler["kind"])
    points, logweight, loglikelihood, derived = sampler.run(
        loglike,
        priors,
        config.sampler.get("settings", {}),
        run_settings=config.sampler.get("run_settings", {}),
        n_derived=len(config.derived_outputs),
    )
    chain = Chain.build(
        priors, points, logweight, loglikelihood,
        derived=derived, derived_names=config.derived_outputs,
    )
    chain.save(config.output["directory"], config.output["name"])
    write_config_with_versions(
        config.raw_text,
        config.output["directory"],
        config.output["name"],
        extra_packages=[*setup.probe.tracked_packages(config), *sampler.TRACKED_PACKAGES],
    )
    return chain


def test_loglike(config: RunConfig) -> float:
    """Evaluate the log-likelihood of a run once, at its fiducial point
    (prior medians for the sampled parameters without a fiducial value).

    Parameters
    ----------
    config: RunConfig
        A pre-loaded running configuration.

    Returns
    -------
    loglike: float
    """
    setup = prepare(config)
    loglike, priors = setup.loglikelihood()
    point = setup.test_point(loglike, priors)
    start = time.perf_counter()
    value = float(setup.probe.loglike(loglike.likelihood, point))
    print(f"loglike at the fiducial point: {value:.6g} ({time.perf_counter() - start:.2f} s)")
    return value


# Not a pytest test.
test_loglike.__test__ = False


def main(argv: list[str] | None = None) -> None:
    """``cloerun chain``: sample the posterior of every selected run."""
    parser = argparse.ArgumentParser(
        prog="cloerun chain", description="Sample the posterior of each run."
    )
    add_run_arguments(parser)
    parser.add_argument(
        "--test", action="store_true",
        help="Only evaluate the log-likelihood at the fiducial point.",
    )
    args = parser.parse_args(argv)
    mode = "test" if args.test else "chain"
    for name, config, _ in load_runs(args.file, args.runs, mode=mode):
        print(f"=== {name} ({mode})")
        test_loglike(config) if args.test else run_chain(config)
