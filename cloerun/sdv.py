"""Synthetic data vectors: the theory at the fiducial point of a run, written
where its config says (``modelling.output.file`` of a pipeline config,
``output.sdv_file`` of a cloerun config), to be read back as its data vector.

    cloerun sdv campaign.yaml --run sdv_WL_fiducial
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cloerun._core.config import RunConfig
from cloerun._core.setup import prepare
from cloerun.campaign import add_run_arguments, load_runs, run_options


def write_sdv(config: RunConfig) -> Path:
    """Write the theory at the fiducial point of a run as a synthetic data
    vector (photometric probes).

    Parameters
    ----------
    config: RunConfig
        A pre-loaded running configuration, with ``output.sdv_file``.

    Returns
    -------
    path: Path
        The written file.
    """
    setup = prepare(config)
    if not hasattr(setup.probe, "theory_spectra"):
        raise ValueError(
            f"Probe '{config.probe['kind']}' cannot write a synthetic data vector; "
            "only the photometric probes can"
        )
    path = Path(config.output["sdv_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    spectra = setup.probe.theory_spectra(setup.likelihood, config, setup.fiducial)
    setup.probe.write_spectra(str(path), spectra)
    print(f"Synthetic data vector written to {path}")
    return path


def main(argv: list[str] | None = None) -> None:
    """``cloerun sdv``: write the synthetic data vector of every selected run."""
    parser = argparse.ArgumentParser(
        prog="cloerun sdv", description="Write the theory at the fiducial point of each run."
    )
    add_run_arguments(parser)
    args = parser.parse_args(argv)
    for name, config, _ in load_runs(args.file, mode="sdv", **run_options(args)):
        print(f"=== {name} (sdv)")
        write_sdv(config)
