"""Reader for the pipeline config: the ``run`` / ``modelling`` / ``inference``
layout of the campaign configs (e.g. those of `dr1-inference-configs`), with
the parameters in a separate file. It is translated into the layout of a
cloerun config yaml file (see `cloerun._core.config.RunConfig`), so both run
through the same code.

Parameter file
--------------
Parameters are grouped in sections (``cosmology``, ``intrinsic_alignment``,
...), whose names do not matter. Each value is either:

- ``value``: fixed;
- ``min start max`` (or ``[min, start, max]``): sampled with a flat prior on
  ``[min, max]``;
- ``gauss mean sigma``: sampled with a Gaussian prior;
- a cloerun spec, e.g. ``{dist: norm, loc: 0.0, scale: 0.01}``: sampled.

``start`` and ``mean`` are the fiducial values, used by the ``sdv`` and
``test`` modes and by the noise-free mock. A name with ``{i}`` is expanded
to every tomographic bin (see `cloerun._core.templates`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from cloerun._core.config import PHOTO_BLOCKS
from cloerun._core.loader import PHOTO_PROBES

SECTIONS = ("run", "modelling", "inference")
# The sdv mode only computes the theory at the fiducial point: it needs no
# data vector, covariance, scale cuts or sampler.
SDV_SECTIONS = ("run", "modelling")

# Names of the parameter file that cloelike reads under another name.
PARAMETER_ALIASES = {"logAs": "logA"}
PARAMETER_PREFIX_ALIASES = {"dz_she_": "dz_shear_"}

# Fixed values of the cloelike parameters a parameter file usually leaves
# out; a value in the file takes precedence.
DEFAULT_FIXED_PARAMETERS = {
    "Omega_k0": 0.0,
    "alpha_s": 0.0,
    "CIA": 0.0134,
    "width_shear_{i}": 1.0,
    "width_pos_{i}": 1.0,
}

# Names of the photometric blocks in ``scale_cuts.per_block``.
BLOCK_ALIASES = {"GC": "GCph"}

_PROBE_BLOCKS = {
    "WL": ("WL",),
    "GCph": ("GCph",),
    "GGL": ("GGL",),
    "2x2pt": ("GGL", "GCph"),
    "3x2pt": ("WL", "GGL", "GCph"),
}


def _parameter_name(name: str) -> str:
    """Name of a parameter of the parameter file as cloelike reads it.

    Parameters
    ----------
    name: str
        Name in the parameter file.

    Returns
    -------
    name: str
    """
    if name in PARAMETER_ALIASES:
        return PARAMETER_ALIASES[name]
    for prefix, alias in PARAMETER_PREFIX_ALIASES.items():
        if name.startswith(prefix):
            return alias + name[len(prefix):]
    return name


def _numbers(name: str, tokens: list[Any]) -> list[float]:
    """Turn the tokens of a parameter value into floats.

    Parameters
    ----------
    name: str
        Parameter name, used in the error message.
    tokens: list[Any]
        Tokens of the value.

    Returns
    -------
    numbers: list[float]
    """
    try:
        return [float(token) for token in tokens]
    except ValueError:
        raise ValueError(
            f"Parameter '{name}': cannot read {tokens} as numbers; expected "
            "'value', 'min start max' or 'gauss mean sigma'"
        ) from None


def parse_parameter(name: str, value: Any) -> tuple[Any, dict[str, Any] | None, Any]:
    """Read one entry of the parameter file.

    Parameters
    ----------
    name: str
        Parameter name, used in the error messages.
    value: Any
        Its value in the parameter file, see the module docstring.

    Returns
    -------
    fixed: Any
        The fixed value, or ``None`` for a sampled parameter.
    prior: dict[str, Any] or None
        The cloerun prior spec of a sampled parameter, or ``None``.
    fiducial: Any
        The fiducial value of a sampled parameter, or ``None``.
    """
    if isinstance(value, dict):
        prior = dict(value)
        fiducial = prior.pop("fiducial", prior.get("loc"))
        return None, prior, fiducial
    if isinstance(value, str):
        tokens = value.replace(",", " ").split()
    elif isinstance(value, list):
        tokens = list(value)
    else:
        return value, None, None

    if tokens and str(tokens[0]).lower() in ("gauss", "norm"):
        numbers = _numbers(name, tokens[1:])
        if len(numbers) != 2:
            raise ValueError(f"Parameter '{name}': expected 'gauss mean sigma', got '{value}'")
        mean, sigma = numbers
        return None, {"dist": "norm", "loc": mean, "scale": sigma}, mean
    numbers = _numbers(name, tokens)
    if len(numbers) == 1:
        return numbers[0], None, None
    if len(numbers) != 3:
        raise ValueError(
            f"Parameter '{name}': expected 'value', 'min start max' or "
            f"'gauss mean sigma', got '{value}'"
        )
    low, start, high = numbers
    if not low <= start <= high:
        raise ValueError(f"Parameter '{name}': expected min <= start <= max, got '{value}'")
    return None, {"dist": "flat", "min": low, "max": high}, start


def parse_parameters(parameters: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Read a parameter file into the cloerun sections.

    Parameters
    ----------
    parameters: dict[str, Any]
        The parsed parameter file: sections of ``name: value`` entries (or
        entries directly at the top level).

    Returns
    -------
    sections: dict[str, dict[str, Any]]
        ``fixed_parameters``, ``sampled_parameters`` and
        ``fiducial_parameters``.
    """
    entries: dict[str, Any] = {}
    for key, value in parameters.items():
        group = value if isinstance(value, dict) and "dist" not in value else {key: value}
        for name, spec in group.items():
            if name in entries:
                raise ValueError(f"Parameter '{name}' is given twice")
            entries[name] = spec

    fixed = dict(DEFAULT_FIXED_PARAMETERS)
    sampled, fiducial = {}, {}
    for name, spec in entries.items():
        value, prior, start = parse_parameter(name, spec)
        name = _parameter_name(name)
        if prior is None:
            fixed[name] = value
            continue
        fixed.pop(name, None)
        sampled[name] = prior
        if start is not None:
            fiducial[name] = start
    return {
        "fixed_parameters": fixed,
        "sampled_parameters": sampled,
        "fiducial_parameters": fiducial,
    }


def _probe_pair_prefixes(probe: str) -> set[tuple[str, str]]:
    """(tracer, tracer) prefixes of the spectra of a probe.

    Parameters
    ----------
    probe: str
        Probe kind.

    Returns
    -------
    prefixes: set[tuple[str, str]]
    """
    return {PHOTO_BLOCKS[block] for block in _PROBE_BLOCKS[probe]}


def _scale_cuts(scale_cuts: dict[str, Any], probe: str, config_dir: Path) -> dict[str, Any]:
    """Translate the ``inference.scale_cuts`` section.

    Parameters
    ----------
    scale_cuts: dict[str, Any]
        ``kind``: ``per_block``, with one range per block under
        ``per_block``, or ``from_file``, with a yaml file of one range per
        spectrum (``SHE-SHE-1-2: [40.0, 1500.0]``) under ``from_file``.
    probe: str
        Probe kind: only the blocks of the probe are kept.
    config_dir: Path
        Directory ``from_file`` is resolved against.

    Returns
    -------
    scale_cuts: dict[str, Any]
        The cloerun ``scale_cuts`` section.
    """
    kind = scale_cuts.get("kind", "per_block")
    blocks = _PROBE_BLOCKS[probe]
    per_block = {
        BLOCK_ALIASES.get(block, block): bounds
        for block, bounds in (scale_cuts.get("per_block") or {}).items()
    }
    if kind == "per_block":
        missing = [block for block in blocks if block not in per_block]
        if missing:
            raise ValueError(f"'scale_cuts.per_block' is missing the range of block(s) {missing}")
        return {block: per_block[block] for block in blocks}
    if kind != "from_file":
        raise ValueError(f"Unknown 'scale_cuts.kind' '{kind}'; expected 'per_block' or 'from_file'")

    path = config_dir / scale_cuts["from_file"]
    pairs = yaml.safe_load(path.read_text()) or {}
    prefixes = _probe_pair_prefixes(probe)
    overrides = {
        pair: bounds for pair, bounds in pairs.items()
        if tuple(str(pair).split("-")[:2]) in prefixes
    }
    # A spectrum missing from the file is an error, unless its block has a
    # range in 'per_block'.
    return {**{block: per_block.get(block) for block in blocks}, "overrides": overrides}


def _resolve(path: str | None, config_dir: Path) -> str | None:
    """Absolute path of a file of the config.

    Parameters
    ----------
    path: str or None
        Path, absolute or relative to ``config_dir``.
    config_dir: Path
        Directory of the config file.

    Returns
    -------
    path: str or None
    """
    return None if path is None else str((config_dir / path).resolve())


def to_run_config(
    config: dict[str, Any], config_dir: Path, name: str, output_dir: str,
    mode: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Translate a pipeline config into a cloerun config.

    Parameters
    ----------
    config: dict[str, Any]
        The pipeline config, with ``modelling.parameters`` already read into
        a dict and every ``${...}`` already replaced (see
        `cloerun.campaign`).
    config_dir: Path
        Directory relative paths are resolved against.
    name: str
        Name of the run, used for the output files.
    output_dir: str
        Directory of the output files.
    mode: str or None
        Overrides ``run.mode`` if given.

    Returns
    -------
    raw: dict[str, Any]
        The cloerun config, to pass to `cloerun._core.config.RunConfig.from_dict`.
    mode: str
        The ``run.mode``: ``chain``, ``sdv`` or ``test``. The ``sdv`` mode
        does not read the ``inference`` section, which may be left out.
    """
    mode = mode or config.get("run", {}).get("mode", "chain")
    sdv = mode == "sdv"
    missing = [
        section for section in (SDV_SECTIONS if sdv else SECTIONS) if section not in config
    ]
    if missing:
        raise ValueError(f"Pipeline config is missing section(s) {missing}")
    modelling = config["modelling"]
    inference = {} if sdv else config["inference"]
    probe = modelling.get("probe")
    if probe not in PHOTO_PROBES:
        raise ValueError(
            f"'modelling.probe' must be one of {sorted(PHOTO_PROBES)}, got '{probe}'"
        )

    theory = modelling.get("theory", {})
    systematics = dict(modelling.get("systematics", {}))
    coupling = modelling.get("coupling", {})
    probe_section = {
        "kind": probe,
        "background": theory["background"],
        "linear_perturbations": theory.get("linear_perturbations", theory["background"]),
        "nonlinear_perturbations": theory["nonlinear_perturbations"],
        "baryon_boost": theory.get("baryon_boost"),
        "mode": coupling.get("mode", "coupled"),
        "settings": {**systematics, **modelling.get("settings", {})},
    }

    nz = modelling["nz"]
    likelihood = inference.get("likelihood", {})
    data: dict[str, Any] = {
        "base_directory": str(config_dir.resolve()),
        "z_grid": {
            key: nz[f"zs_{key}"] for key in ("min", "max", "num") if f"zs_{key}" in nz
        },
        "noise_free_mock": bool(likelihood.get("noise_free_mock", False)),
    }
    if "file" in nz:
        data["nz"] = nz["file"]
    for key in ("shear", "positions"):
        if key in nz:
            data[f"nz_{key}"] = nz[key]
    for key, source in (
        ("cells", likelihood.get("data_vector")),
        ("covariance", None if sdv else likelihood["covariance"]),
        ("covariance_key", likelihood.get("covariance_key")),
        ("mixing", coupling.get("mixing_matrices") if probe_section["mode"] == "coupled" else None),
    ):
        if source is not None:
            data[key] = source
    if "binning" in modelling:
        data["ells"] = dict(modelling["binning"])

    output = modelling.get("output", {})
    raw = {
        "probe": probe_section,
        "data": data,
        **parse_parameters(modelling["parameters"]),
        "prune_parameters": True,
        "derived_outputs": list(inference.get("derived_outputs", [])),
        "output": {
            "directory": output_dir,
            "name": name,
            "sdv_file": _resolve(output.get("file", f"{output_dir}/{name}_sdv.fits"), config_dir),
        },
    }
    if not sdv:
        raw["scale_cuts"] = _scale_cuts(inference["scale_cuts"], probe, config_dir)
        raw["sampler"] = inference["sampler"]
    return raw, mode


_VARIABLE = re.compile(r"\$\{(\w+)\}")


def interpolate(value: Any, variables: dict[str, str]) -> Any:
    """Replace every ``${name}`` in the strings of ``value``.

    Parameters
    ----------
    value: Any
        A config, or any part of it.
    variables: dict[str, str]
        Values of the variables.

    Returns
    -------
    value: Any
        Same structure, with the variables replaced.
    """
    if isinstance(value, dict):
        return {key: interpolate(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [interpolate(item, variables) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match) -> str:
        if match.group(1) not in variables:
            raise ValueError(
                f"Unknown variable '${{{match.group(1)}}}' in '{value}'; "
                f"available: {sorted(variables)}"
            )
        return str(variables[match.group(1)])

    return _VARIABLE.sub(replace, value)
