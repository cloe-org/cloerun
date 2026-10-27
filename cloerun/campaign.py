"""Campaigns: sets of runs sharing a baseline, run with
``cloerun campaign campaign.yaml``.

A campaign file lists the runs, each either built from a pipeline config
(``base``, see `cloerun._core.pipeline`) or from another run (``parent``), and
modified by ``components``::

    output_dir:  /path/to/output/directory
    output_name: campaign_scalecuts
    runs:
      - name: chain_WL_baseline
        base: config.yaml
      - name: chain_WL_baseline_w0wa
        parent: chain_WL_baseline
        components: [priors_w0wa]
    components:
      priors_w0wa:
        w0: -3.0 -1.0 -0.33

Each key of a component is merged into the config where that key is: at the
top level (``run``, ``modelling``, ``inference``), else at the one place in
the config, parameters included, where it appears (``scale_cuts``, ``w0``,
...). A dotted key (``inference.likelihood.covariance_key``) gives the place
explicitly; it is needed when the key appears in several places (``mode``,
``file``, ...).

The strings of the config may use ``${name}`` (the run name),
``${output_dir}`` (``<output_dir>/<output_name>``, where the outputs of the
runs are written) and ``${root_dir}`` (``run.root_dir``). Relative paths are
resolved against the directory of the campaign file for ``base`` and
``output_dir``, and against the directory of the pipeline config otherwise.

A pipeline config can also be run on its own: ``cloerun campaign config.yaml``.
The ``cloerun sdv``, ``chain`` and ``analyse`` commands take the same files.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cloerun._core.config import RunConfig
from cloerun._core.pipeline import interpolate, to_run_config


@dataclass
class CampaignRun:
    """One resolved run of a campaign.

    Attributes
    ----------
    name: str
        Name of the run.
    config: dict[str, Any]
        The pipeline config of the run, components applied and variables
        replaced.
    config_dir: Path
        Directory of its pipeline config file.
    output_dir: str
        Directory of its output files.
    """

    name: str
    config: dict[str, Any]
    config_dir: Path
    output_dir: str

    def to_run_config(self, mode: str | None = None) -> tuple[RunConfig, str]:
        """The cloerun config of the run.

        Parameters
        ----------
        mode: str or None
            Overrides ``run.mode`` if given.

        Returns
        -------
        config: RunConfig
        mode: str
        """
        raw, mode = to_run_config(
            self.config, self.config_dir, self.name, self.output_dir, mode
        )
        return RunConfig.from_dict(raw, self.config_dir, label=self.name), mode


def _deep_merge(base: Any, update: Any) -> Any:
    """Merge ``update`` into ``base``: dicts are merged key by key, anything
    else is replaced.

    Parameters
    ----------
    base: Any
    update: Any

    Returns
    -------
    merged: Any
    """
    if not (isinstance(base, dict) and isinstance(update, dict)):
        return copy.deepcopy(update)
    merged = dict(base)
    for key, value in update.items():
        merged[key] = _deep_merge(base[key], value) if key in base else copy.deepcopy(value)
    return merged


def _find_key(tree: dict[str, Any], key: str, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Every path at which ``key`` appears in a nested dict.

    Parameters
    ----------
    tree: dict[str, Any]
    key: str
    path: tuple[str, ...]
        Path of ``tree`` itself.

    Returns
    -------
    paths: list[tuple[str, ...]]
    """
    paths = []
    for name, value in tree.items():
        if name == key:
            paths.append((*path, name))
        if isinstance(value, dict):
            paths.extend(_find_key(value, key, (*path, name)))
    return paths


def apply_component(config: dict[str, Any], component: dict[str, Any], label: str) -> dict[str, Any]:
    """Apply one component to a pipeline config.

    Parameters
    ----------
    config: dict[str, Any]
        The pipeline config, with ``modelling.parameters`` already read.
    component: dict[str, Any]
        The component, see the module docstring.
    label: str
        Name of the component, used in the error messages.

    Returns
    -------
    config: dict[str, Any]
        A new config with the component applied.
    """
    config = copy.deepcopy(config)
    for key, value in component.items():
        if "." in key:
            path = tuple(key.split("."))
        elif key in config:
            path = (key,)
        else:
            paths = _find_key(config, key)
            if not paths:
                raise ValueError(f"Component '{label}': '{key}' is not in the config")
            if len(paths) > 1:
                options = [".".join(p) for p in paths]
                raise ValueError(
                    f"Component '{label}': '{key}' is ambiguous, write one of {options}"
                )
            path = paths[0]
        parent = config
        for name in path[:-1]:
            parent = parent.setdefault(name, {})
        parent[path[-1]] = _deep_merge(parent.get(path[-1]), value)
    return config


def load_pipeline_config(path: Path) -> dict[str, Any]:
    """Read a pipeline config and its parameter file.

    Parameters
    ----------
    path: Path
        The pipeline config file.

    Returns
    -------
    config: dict[str, Any]
        The config, with ``modelling.parameters`` read into a dict.
    """
    config = yaml.safe_load(path.read_text())
    parameters = config.get("modelling", {}).get("parameters")
    if isinstance(parameters, str):
        config["modelling"]["parameters"] = yaml.safe_load(
            (path.parent / parameters).read_text()
        )
    return config


def _variables(config: dict[str, Any], name: str, output_dir: str) -> dict[str, str]:
    """Values of the ``${...}`` variables of a run.

    Parameters
    ----------
    config: dict[str, Any]
        The pipeline config of the run.
    name: str
        Name of the run.
    output_dir: str
        Directory of the outputs of the run.

    Returns
    -------
    variables: dict[str, str]
    """
    variables = {"name": name, "output_dir": output_dir}
    root_dir = config.get("run", {}).get("root_dir")
    if root_dir is not None:
        variables["root_dir"] = str(root_dir)
    return variables


def load_campaign(path: str | Path) -> list[CampaignRun]:
    """Resolve every run of a campaign file, or the single run of a
    pipeline config file.

    Parameters
    ----------
    path: str or Path
        The campaign (or pipeline config) file.

    Returns
    -------
    runs: list[CampaignRun]
        In the order of the campaign file.
    """
    path = Path(path).resolve()
    campaign = yaml.safe_load(path.read_text())
    if "runs" not in campaign:
        config = load_pipeline_config(path)
        name = config.get("run", {}).get("name", path.stem)
        output_dir = str(path.parent / config.get("run", {}).get("output_dir", "chains"))
        config = interpolate(config, _variables(config, name, output_dir))
        return [CampaignRun(name, config, path.parent, output_dir)]

    for key in ("output_dir", "output_name"):
        if key not in campaign:
            raise ValueError(f"Campaign '{path}' is missing '{key}'")
    output_dir = str(path.parent / campaign["output_dir"] / campaign["output_name"])
    components = campaign.get("components") or {}
    specs = {}
    for spec in campaign["runs"]:
        if spec["name"] in specs:
            raise ValueError(f"Run '{spec['name']}' is defined twice")
        specs[spec["name"]] = spec

    # Configs before interpolation, so that children can still override what
    # the variables depend on (e.g. run.root_dir).
    resolved: dict[str, tuple[dict[str, Any], Path]] = {}

    def resolve(name: str, chain: tuple[str, ...] = ()) -> tuple[dict[str, Any], Path]:
        if name in chain:
            raise ValueError(f"Runs {[*chain, name]} are their own parent")
        if name not in specs:
            raise ValueError(f"Unknown run '{name}'")
        if name in resolved:
            return resolved[name]
        spec = specs[name]
        if ("base" in spec) == ("parent" in spec):
            raise ValueError(f"Run '{name}' needs exactly one of 'base' and 'parent'")
        if "base" in spec:
            base = path.parent / spec["base"]
            config, config_dir = load_pipeline_config(base), base.parent
        else:
            config, config_dir = resolve(spec["parent"], (*chain, name))
        for label in spec.get("components") or []:
            if label not in components:
                raise ValueError(f"Run '{name}': unknown component '{label}'")
            config = apply_component(config, components[label], label)
        resolved[name] = config, config_dir
        return resolved[name]

    runs = []
    for name in specs:
        config, config_dir = resolve(name)
        config = interpolate(config, _variables(config, name, output_dir))
        runs.append(CampaignRun(name, config, config_dir, output_dir))
    return runs


MODES = ("chain", "sdv", "test")


def load_runs(
    path: str | Path, names: list[str] | None = None, mode: str | None = None
) -> list[tuple[str, RunConfig, str]]:
    """The runs of a campaign file, of a pipeline config, or of a cloerun
    config (``probe``, ``data``, ... at the top level).

    Parameters
    ----------
    path: str or Path
        The config file.
    names: list[str] or None
        Only these runs, in the order of the file; every run if ``None``.
    mode: str or None
        Overrides ``run.mode`` if given.

    Returns
    -------
    runs: list[tuple[str, RunConfig, str]]
        Name, config and mode of each run.
    """
    if "probe" in yaml.safe_load(Path(path).read_text()):
        config = RunConfig.from_yaml(str(path))
        runs = [(config.output["name"], config, mode or "chain")]
    else:
        runs = [(r.name, *r.to_run_config(mode)) for r in load_campaign(path)]
    if names:
        unknown = sorted(set(names) - {name for name, _, _ in runs})
        if unknown:
            raise ValueError(
                f"Unknown run(s) {unknown}; available: {[name for name, _, _ in runs]}"
            )
        runs = [run for run in runs if run[0] in names]
    return runs


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by the ``cloerun`` commands: the config file and the
    selection of runs.

    Parameters
    ----------
    parser: argparse.ArgumentParser
    """
    parser.add_argument("file", help="Campaign file, pipeline config or cloerun config.")
    parser.add_argument(
        "--run", action="append", dest="runs", metavar="NAME",
        help="Only this run (repeat for several). Default: every run.",
    )


def run_campaign(
    path: str | Path, names: list[str] | None = None, mode: str | None = None
) -> dict[str, Any]:
    """Run every selected run of a campaign in its ``run.mode``.

    Parameters
    ----------
    path: str or Path
        The config file, see `load_runs`.
    names: list[str] or None
        Only these runs.
    mode: str or None
        Overrides ``run.mode`` if given.

    Returns
    -------
    results: dict[str, Any]
        By run: the `cloerun.chain.Chain` (``chain``), the path of the
        synthetic data vector (``sdv``) or the log-likelihood (``test``).
    """
    from cloerun.chain import run_chain, test_loglike
    from cloerun.sdv import write_sdv

    actions = {"chain": run_chain, "sdv": write_sdv, "test": test_loglike}
    results = {}
    for name, config, run_mode in load_runs(path, names, mode):
        if run_mode not in actions:
            raise ValueError(
                f"Run '{name}': unknown mode '{run_mode}'; expected one of {list(MODES)}"
            )
        print(f"=== {name} ({run_mode})")
        results[name] = actions[run_mode](config)
    return results


def main(argv: list[str] | None = None) -> None:
    """``cloerun campaign``: run every run of a campaign in its ``run.mode``."""
    parser = argparse.ArgumentParser(
        prog="cloerun campaign",
        description="Run each run of a campaign in its 'run.mode' (chain, sdv or test).",
    )
    add_run_arguments(parser)
    parser.add_argument("--mode", choices=MODES, help="Override 'run.mode' of every run.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolved cloerun config of each run, without running it.",
    )
    args = parser.parse_args(argv)
    if not args.dry_run:
        run_campaign(args.file, args.runs, args.mode)
        return
    for name, config, mode in load_runs(args.file, args.runs, args.mode):
        print(f"=== {name} ({mode})")
        print(config.raw_text)
