"""Analysis of finished chains: a summary table of the posterior and a
triangle plot (with `getdist`), written next to the chain.

    cloerun analyse campaign.yaml --run chain_WL_baseline
    cloerun analyse campaign.yaml --params omch2 H0 S8

For a run named ``<name>`` this writes ``<name>.summary.md`` and
``<name>.triangle.pdf`` to its output directory.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from cloerun._core.chain_data import Chain
from cloerun._core.config import RunConfig
from cloerun._core.templates import expand_bin_templates, prune_unused
from cloerun.campaign import add_run_arguments, load_runs, run_options

# Axis labels of the usual parameters; others are labelled with their name.
LABELS = {
    "omch2": r"\omega_{\rm c}",
    "ombh2": r"\omega_{\rm b}",
    "Omega_cdm0": r"\Omega_{\rm c}",
    "Omega_b0": r"\Omega_{\rm b}",
    "logA": r"\ln(10^{10}A_{\rm s})",
    "As": r"A_{\rm s}",
    "ns": r"n_{\rm s}",
    "H0": r"H_0",
    "h": r"h",
    "w0": r"w_0",
    "wa": r"w_a",
    "sigma8_0": r"\sigma_8",
    "S8": r"S_8",
    "AIA": r"A_{\rm IA}",
    "EtaIA": r"\eta_{\rm IA}",
    "log10TAGN": r"\log_{10} T_{\rm AGN}",
}

# Parameters with one value per tomographic bin, left out of the default plot.
_PER_BIN = re.compile(r".*(_\d+|bin\d+)$")


@dataclass
class Summary:
    """Posterior summary of one parameter.

    Attributes
    ----------
    name: str
    mean: float
        Weighted posterior mean.
    std: float
        Weighted posterior standard deviation.
    low: float
        16th percentile.
    median: float
    high: float
        84th percentile.
    fiducial: float or None
        Fiducial value of the run, if known.
    """

    name: str
    mean: float
    std: float
    low: float
    median: float
    high: float
    fiducial: float | None = None

    @property
    def shift(self) -> float | None:
        """Distance of the mean from the fiducial, in posterior sigmas."""
        if self.fiducial is None or self.std == 0:
            return None
        return (self.mean - self.fiducial) / self.std


def weights(chain: Chain) -> np.ndarray:
    """Normalised posterior weight of each sample.

    Parameters
    ----------
    chain: Chain

    Returns
    -------
    weights: np.ndarray
    """
    w = np.exp(chain.logweight - chain.logweight.max())
    return w / w.sum()


def _quantiles(values: np.ndarray, w: np.ndarray, q: Sequence[float]) -> np.ndarray:
    """Weighted quantiles.

    Parameters
    ----------
    values: np.ndarray
    w: np.ndarray
        Normalised weights.
    q: Sequence[float]
        Quantiles, in [0, 1].

    Returns
    -------
    quantiles: np.ndarray
    """
    order = np.argsort(values)
    cdf = np.cumsum(w[order]) - 0.5 * w[order]
    return np.interp(q, cdf, values[order])


def fiducial_values(config: RunConfig, chain: Chain) -> dict[str, float]:
    """Fiducial value of each chain column, where the config gives one.

    Parameters
    ----------
    config: RunConfig
        The run's configuration.
    chain: Chain

    Returns
    -------
    fiducial: dict[str, float]
    """
    section = {**config.fixed_parameters, **config.fiducial_parameters}
    # Expand the '{i}' templates against the chain columns: no likelihood needed.
    section, _ = prune_unused(section, chain.parameters)
    expanded = expand_bin_templates(section, chain.parameters, "fiducial_parameters")
    return {
        name: float(value) for name, value in expanded.items()
        if name in chain.parameters and np.isscalar(value)
    }


def summarise(chain: Chain, fiducial: dict[str, float] | None = None) -> list[Summary]:
    """Posterior summary of every parameter of a chain.

    Parameters
    ----------
    chain: Chain
    fiducial: dict[str, float] or None
        Fiducial values, see `fiducial_values`.

    Returns
    -------
    summaries: list[Summary]
        In the order of the chain columns.
    """
    fiducial = fiducial or {}
    w = weights(chain)
    summaries = []
    for name in chain.parameters:
        values = chain.column(name)
        mean = float(np.sum(w * values))
        std = float(np.sqrt(np.sum(w * (values - mean) ** 2)))
        low, median, high = _quantiles(values, w, [0.16, 0.5, 0.84])
        summaries.append(
            Summary(name, mean, std, float(low), float(median), float(high), fiducial.get(name))
        )
    return summaries


def summary_table(summaries: list[Summary], title: str = "") -> str:
    """Markdown table of the posterior summaries.

    Parameters
    ----------
    summaries: list[Summary]
    title: str
        Heading of the table.

    Returns
    -------
    table: str
    """
    def fmt(value: float | None) -> str:
        return "" if value is None else f"{value:.5g}"

    lines = [f"# {title}", ""] if title else []
    lines += [
        "| parameter | fiducial | mean | std | 16% | median | 84% | (mean - fid) / std |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        shift = "" if s.shift is None else f"{s.shift:+.2f}"
        lines.append(
            f"| {s.name} | {fmt(s.fiducial)} | {fmt(s.mean)} | {fmt(s.std)} | "
            f"{fmt(s.low)} | {fmt(s.median)} | {fmt(s.high)} | {shift} |"
        )
    return "\n".join(lines) + "\n"


def default_plot_parameters(chain: Chain) -> list[str]:
    """Parameters of the default triangle plot: every column but the
    per-bin nuisance parameters.

    Parameters
    ----------
    chain: Chain

    Returns
    -------
    names: list[str]
    """
    return [name for name in chain.parameters if not _PER_BIN.fullmatch(name)]


def triangle_plot(
    chain: Chain,
    path: str | Path,
    parameters: Sequence[str] | None = None,
    fiducial: dict[str, float] | None = None,
    label: str | None = None,
) -> Path:
    """Triangle plot of a chain, with the fiducial values marked.

    Parameters
    ----------
    chain: Chain
    path: str or Path
        Output file (``.pdf``, ``.png``, ...).
    parameters: Sequence[str] or None
        Parameters to plot; `default_plot_parameters` if ``None``.
    fiducial: dict[str, float] or None
        Fiducial values to mark.
    label: str or None
        Legend label.

    Returns
    -------
    path: Path
    """
    import matplotlib

    matplotlib.use("Agg")
    from getdist import MCSamples, plots

    parameters = list(parameters or default_plot_parameters(chain))
    unknown = [name for name in parameters if name not in chain.parameters]
    if unknown:
        raise ValueError(f"Not in the chain: {unknown}; available: {chain.parameters}")
    samples = MCSamples(
        samples=np.column_stack([chain.column(name) for name in parameters]),
        weights=weights(chain),
        names=parameters,
        labels=[LABELS.get(name, name.replace("_", r"\_")) for name in parameters],
        label=label,
        settings={"ignore_rows": 0},
    )
    plotter = plots.get_subplot_plotter()
    markers = {k: v for k, v in (fiducial or {}).items() if k in parameters}
    plotter.triangle_plot(samples, parameters, filled=True, markers=markers)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.export(str(path))
    return path


def analyse_run(config: RunConfig, parameters: Sequence[str] | None = None) -> list[Summary]:
    """Summary table and triangle plot of a finished run, written next to
    its chain.

    Parameters
    ----------
    config: RunConfig
        The run's configuration.
    parameters: Sequence[str] or None
        Parameters of the triangle plot.

    Returns
    -------
    summaries: list[Summary]
    """
    directory, name = Path(config.output["directory"]), config.output["name"]
    chain = Chain.load(directory / f"{name}.npz")
    fiducial = fiducial_values(config, chain)
    summaries = summarise(chain, fiducial)
    table = directory / f"{name}.summary.md"
    table.write_text(summary_table(summaries, title=name))
    plot = triangle_plot(
        chain, directory / f"{name}.triangle.pdf", parameters, fiducial, label=name
    )
    print(f"Summary written to {table}\nTriangle plot written to {plot}")
    return summaries


def main(argv: list[str] | None = None) -> None:
    """``cloerun analyse``: summarise and plot every selected run."""
    parser = argparse.ArgumentParser(
        prog="cloerun analyse", description="Summary table and triangle plot of each run."
    )
    add_run_arguments(parser)
    parser.add_argument(
        "--params", nargs="+", metavar="NAME",
        help="Parameters of the triangle plot. Default: all but the per-bin ones.",
    )
    args = parser.parse_args(argv)
    for name, config, _ in load_runs(args.file, mode="chain", **run_options(args)):
        print(f"=== {name} (analyse)")
        analyse_run(config, args.params)
