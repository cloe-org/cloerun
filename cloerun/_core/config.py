"""Main reader for a cloerun config yaml file."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from cloerun._core.loader import DERIVED_PARAMETERS, PHOTO_PROBES

REQUIRED_SECTIONS = (
    "probe", "data", "fixed_parameters", "sampled_parameters", "sampler", "output",
)
# The sdv mode only computes the theory at the fiducial point: it needs no
# sampler (nor covariance or scale cuts).
SDV_REQUIRED_SECTIONS = tuple(s for s in REQUIRED_SECTIONS if s != "sampler")


def _redshift_label(z: float) -> str:
    """Label a redshift according to a naming convention (with no trailing zeros).

    Parameters
    ----------
    z: float
        Redshift value.

    Returns
    -------
    label: str
        Literal string representing the redshift, with no trailing zeros on the right.
    """
    return str(z).rstrip("0")


def _check_redshift_keys_match(keys: set, data: DataConfig, label: str) -> None:
    """Raise an error if ``keys`` does not coincide with the expected redshift keys.

    Parameters
    ----------
    keys: set
        Keys specified in a config section/spec.
    data: DataConfig
        The resolved ``data`` section.
    label: str
        Name of the section/spec, used in the error message.
    """
    expected = set(data.redshifts)
    if keys != expected:
        missing = sorted(expected - keys)
        unexpected = sorted(keys - expected)
        parts = []
        if missing: parts.append(f"missing {missing}")
        if unexpected: parts.append(f"unexpected {unexpected}")
        detail = ", ".join(parts)
        raise ValueError(
            f"'{label}' keys must match 'data.redshifts' {sorted(expected)} "
            f"exactly: {detail}"
        )


def _parse_scale_cuts(scale_cuts: dict[float, Any], data: DataConfig) -> dict[str, Any]:
    """Re-key the raw ``scale_cuts`` section using data labels rather than redshifts.

    Parameters
    ----------
    scale_cuts: dict[float, Any]
        The raw ``scale_cuts`` section of a config yaml file.
    data: DataConfig
        The resolved ``data`` section.

    Returns
    -------
    scale_cuts: dict[str, Any]
        Re-keyed ``scale_cuts`` section, using data labels instead of redshifts.
    """
    _check_redshift_keys_match(set(scale_cuts), data, "scale_cuts")
    label = dict(zip(data.redshifts, data.labels))
    return {label[z]: bounds for z, bounds in scale_cuts.items()}


# Blocks of the photometric data vector, as named in the ``scale_cuts`` section,
# mapped to the (tracer, tracer) prefix of the corresponding data keys.
PHOTO_BLOCKS = {
    "WL": ("SHE", "SHE"),
    "GGL": ("POS", "SHE"),
    "GCph": ("POS", "POS"),
}


def _parse_photo_scale_cuts(scale_cuts: dict[str, Any]) -> dict[str, Any]:
    """Validate the photometric ``scale_cuts`` section and parse the keys of
    its per-pair ``overrides``.

    Parameters
    ----------
    scale_cuts: dict[str, Any]
        The raw ``scale_cuts`` section of a config yaml file, with one
        multipole range per block (``WL``, ``GGL``, ``GCph``) and optional
        per-pair ranges under ``overrides``, keyed as ``"POS-SHE-1-3"``. A
        block range may be ``null`` if ``overrides`` covers every spectrum
        of the block.

    Returns
    -------
    scale_cuts: dict[str, Any]
        Same section, with the ``overrides`` keys turned into data keys,
        e.g. ``("POS", "SHE", 1, 3)``.
    """
    unexpected = sorted(set(scale_cuts) - set(PHOTO_BLOCKS) - {"overrides"})
    if unexpected:
        raise ValueError(
            f"Unexpected 'scale_cuts' key(s) {unexpected}; expected any of "
            f"{sorted(PHOTO_BLOCKS)} and 'overrides'"
        )
    parsed = {
        block: None if bounds is None else list(bounds)
        for block, bounds in scale_cuts.items() if block != "overrides"
    }
    overrides = {}
    for pair, bounds in scale_cuts.get("overrides", {}).items():
        parts = str(pair).split("-")
        if len(parts) != 4 or tuple(parts[:2]) not in PHOTO_BLOCKS.values():
            raise ValueError(
                f"Invalid 'scale_cuts.overrides' key '{pair}'; expected e.g. "
                "'SHE-SHE-1-2', 'POS-SHE-1-3' or 'POS-POS-2-2'"
            )
        overrides[(parts[0], parts[1], int(parts[2]), int(parts[3]))] = list(bounds)
    parsed["overrides"] = overrides
    return parsed


def _check_photo_sampled_parameters(sampled_parameters: dict[str, Any]) -> None:
    """Raise an error for sampled-parameter specs the photometric probes do
    not support: per-redshift specs and analytical marginalisation.

    Parameters
    ----------
    sampled_parameters: dict[str, Any]
        The raw ``sampled_parameters`` section of a config yaml file.
    """
    for name, spec in sampled_parameters.items():
        if not isinstance(spec, dict) or "dist" not in spec:
            raise ValueError(
                f"'sampled_parameters.{name}' must be a single spec with a 'dist' "
                "key for photometric probes; use a '{i}' template (e.g. "
                "'dz_shear_{i}') for per-bin parameters"
            )
        if spec["dist"] == "AM":
            raise ValueError(
                f"'sampled_parameters.{name}': analytical marginalisation "
                "(dist: AM) is not available for photometric probes"
            )


def _normalize_sampled_parameters(sampled_parameters: dict[str, Any], data: DataConfig) -> dict[str, Any]:
    """Turn per-bin sampled parameters into ordered list, according to data.redshifts.

    Parameters
    ----------
    sampled_parameters: dict[str, Any]
        The raw ``sampled_parameters`` section of a config yaml file.
    data: DataConfig
        The resolved ``data`` section.

    Returns
    -------
    normalized: dict[str, Any]
        Dictionary containing the sampled parameters arranged in lists for nuisance parameters with
        different priors per redshift.
    """
    normalized = {}
    for name, spec in sampled_parameters.items():
        if isinstance(spec, dict) and "dist" not in spec:
            _check_redshift_keys_match(set(spec), data, f"sampled_parameters.{name}")
            spec = [spec[z] for z in data.redshifts]
        normalized[name] = spec
    return normalized


def _auto_enable_derived_parameters(
    derived_parameters: dict[str, bool],
    fixed_parameters: dict[str, Any],
    sampled_parameters: dict[str, Any],
) -> dict[str, bool]:
    """Extend the raw ``derived_parameters`` section with any auto-detected
    relations whose trigger is present in the fixed or sampled parameters.

    Parameters
    ----------
    derived_parameters: dict[str, bool]
        The raw ``derived_parameters`` section of a config yaml file.
    fixed_parameters: dict[str, Any]
        The raw ``fixed_parameters`` section.
    sampled_parameters: dict[str, Any]
        The resolved ``sampled_parameters`` section.

    Returns
    -------
    derived_parameters: dict[str, bool]
        Updated set of derived parameters after auto-detection.
    """
    present = set(fixed_parameters) | set(sampled_parameters)
    auto_enabled = {
        name: True
        for name, relation in DERIVED_PARAMETERS.items()
        if relation.get("trigger") in present
    }
    return {**auto_enabled, **derived_parameters}


@dataclass
class DataConfig:
    """Container for data setup.

    Attributes
    ----------
    redshifts: list[float]
        Redshift values.
    labels: list[str]
        Corresponding redshift labels, with no trailing zeros.
    bao_labels: list[str]
        Corresponding redshift labels, with trailing zeros.
    datavec_FS_template: str or None
        Full-shape (power-spectrum multipoles) data-vector filename
        template, resolved to an absolute path, or ``None`` if not
        given in the config.
    datavec_BAO_template: str or None
        BAO data-vector filename template, resolved to an absolute
        path, or ``None`` if not given in the config.
    covariance_template: str
        Covariance filename template, resolved to an absolute path -- always
        required.
    mixing_template: str or None
        Mixing-matrix filename template, resolved to an absolute path, or
        ``None`` if not given in the config.
    num_mocks: int or None
        Number of mocks used to estimate a numerical covariance, which
        triggers the Hartlap correction, or ``None`` for an analytical one.
    """

    redshifts: list[float]
    labels: list[str]
    bao_labels: list[str]
    datavec_FS_template: Optional[str]
    datavec_BAO_template: Optional[str]
    covariance_template: str
    mixing_template: Optional[str]
    num_mocks: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict, config_dir: Path) -> DataConfig:
        """Build a `DataConfig` object from the raw ``data`` section of a config yaml file.

        Parameters
        ----------
        data: dict
            The ``data`` section parsed from a config yaml file.
        config_dir: Path
            Directory the config file lives in.

        Returns
        -------
        data_config: DataConfig
        """
        for key in ("redshifts", "covariance"):
            if key not in data:
                raise ValueError(f"'data' section is missing required key '{key}'")

        base_directory = config_dir / data.get("base_directory", ".")
        redshifts = list(data["redshifts"])
        labels = [_redshift_label(z) for z in redshifts]
        bao_labels = [str(z) for z in redshifts]

        return cls(
            redshifts=redshifts,
            labels=labels,
            bao_labels=bao_labels,
            datavec_FS_template=(
                str(base_directory / data["datavec_FS"]) if "datavec_FS" in data else None
            ),
            datavec_BAO_template=(
                str(base_directory / data["datavec_BAO"]) if "datavec_BAO" in data else None
            ),
            covariance_template=str(base_directory / data["covariance"]),
            mixing_template=str(base_directory / data["mixing"]) if "mixing" in data else None,
            num_mocks=data.get("num_mocks"),
        )


@dataclass
class PhotoDataConfig:
    """Container for the data setup of the photometric probes.

    Attributes
    ----------
    nz_positions: str
        Path of the n(z) file of the positions (lens) sample.
    nz_shear: str
        Path of the n(z) file of the shear (source) sample.
    z_grid: dict[str, float]
        ``min``, ``max`` and ``num`` of the uniform redshift grid the n(z)
        are resampled on, which is also the grid of the theory calculation.
    cells: str or None
        Path of the angular power spectra file, or ``None`` with
        ``noise_free_mock: true``: the data vector is then the theory at the
        fiducial, on the multipoles given by ``ells``.
    covariance: str or None
        Path of the covariance file (``.npz`` or ``.npy``). It covers either
        exactly the blocks of the probe, or the full 3x2pt data vector, in
        which case the blocks of the probe are extracted from it. ``None``
        only in the ``sdv`` mode, which does not need it.
    covariance_key: str or None
        Array to read from a ``.npz`` covariance file with several arrays.
    mixing: str or None
        Path of the mixing matrices file, required for ``probe.mode:
        coupled``.
    noise_free_mock: bool
        Whether to replace the data vector with the theory at
        ``fixed_parameters``.
    ells: dict[str, Any] or None
        Multipole binning used when there is no ``cells`` file, see
        `cloerun._core.probes.photo_cls.binning_ells`.
    """

    nz_positions: str
    nz_shear: str
    z_grid: dict[str, float]
    cells: Optional[str]
    covariance: Optional[str]
    covariance_key: Optional[str] = None
    mixing: Optional[str] = None
    noise_free_mock: bool = False
    ells: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict, config_dir: Path, sdv: bool = False) -> PhotoDataConfig:
        """Build a `PhotoDataConfig` object from the raw ``data`` section of a
        config yaml file.

        Parameters
        ----------
        data: dict
            The ``data`` section parsed from a config yaml file.
        config_dir: Path
            Directory the config file lives in.
        sdv: bool
            Whether the run is in the ``sdv`` mode, where ``covariance`` is
            optional.

        Returns
        -------
        data_config: PhotoDataConfig
        """
        if "covariance" not in data and not sdv:
            raise ValueError("'data' section is missing required key 'covariance'")
        if "cells" not in data and "ells" not in data:
            raise ValueError("'data' section needs 'cells', or 'ells' to build the data vector")
        nz_positions = data.get("nz_positions", data.get("nz"))
        nz_shear = data.get("nz_shear", data.get("nz"))
        if nz_positions is None or nz_shear is None:
            raise ValueError(
                "'data' section needs 'nz' (shared by positions and shear) or "
                "both 'nz_positions' and 'nz_shear'"
            )

        base_directory = config_dir / data.get("base_directory", ".")
        z_grid = {"min": 1e-4, "max": 3.0, "num": 100, **data.get("z_grid", {})}
        ells = dict(data["ells"]) if "ells" in data else None
        if ells is not None and "file" in ells:
            ells["file"] = str(base_directory / ells["file"])

        return cls(
            nz_positions=str(base_directory / nz_positions),
            nz_shear=str(base_directory / nz_shear),
            z_grid=z_grid,
            cells=str(base_directory / data["cells"]) if "cells" in data else None,
            covariance=(
                str(base_directory / data["covariance"]) if "covariance" in data else None
            ),
            covariance_key=data.get("covariance_key"),
            mixing=str(base_directory / data["mixing"]) if "mixing" in data else None,
            noise_free_mock=bool(data.get("noise_free_mock", False)),
            ells=ells,
        )


@dataclass
class RunConfig:
    """Container for full setup, including kind of probe, data,
       scale cuts, priors, sampler, and output settings.

    Attributes
    ----------
    probe: dict[str, Any]
        The raw ``probe`` section of the config yaml file.
    data: DataConfig or PhotoDataConfig
        The resolved ``data`` section of the config yaml file, in the
        spectroscopic or photometric layout depending on ``probe.kind``.
    scale_cuts: dict[str, Any]
        The ``scale_cuts`` section of the config yaml file: re-keyed from
        redshift value to redshift label for the spectroscopic probes, or
        with parsed ``overrides`` keys for the photometric ones.
    sampled_parameters: dict[str, Any]
        The ``sampled_parameters`` section of the config yaml file, processed
        to include per-bin entries (spectroscopic probes). Photometric
        ``{i}`` templates are expanded later, once the likelihood is built.
    fixed_parameters: dict[str, Any]
        The raw ``fixed_parameters`` section of the config yaml file.
    sampler: dict[str, Any]
        The raw ``sampler`` section of the config yaml file.
    output: dict[str, Any]
        The raw ``output`` section of the config yaml file.
    raw_text: str
        The original config yaml file in a string format. Warning: this does not
        track any changes made to the config object after loading.
    constraints: dict[str, bool]
        The raw ``constraints`` section of the config yaml file.
    derived_parameters: dict[str, bool]
        The ``derived_parameters`` section of the config yaml file.
    derived_outputs: list[str]
        The ``derived_outputs`` section: quantities computed at each sample
        and stored as extra chain columns (photometric probes:
        ``sigma8_0``, ``S8``).
    fiducial_parameters: dict[str, Any]
        The ``fiducial_parameters`` section: values of the sampled parameters
        at the fiducial point, used with ``fixed_parameters`` by the ``sdv``
        and ``test`` modes.
    prune_parameters: bool
        Whether to drop the parameters the likelihood does not read, instead
        of failing on them -- for parameter files shared by several probes.
    """

    probe: dict[str, Any]
    data: DataConfig | PhotoDataConfig
    scale_cuts: dict[str, Any]
    sampled_parameters: dict[str, Any]
    fixed_parameters: dict[str, Any]
    sampler: dict[str, Any]
    output: dict[str, Any]
    raw_text: str
    constraints: dict[str, bool] = field(default_factory=dict)
    derived_parameters: dict[str, bool] = field(default_factory=dict)
    derived_outputs: list[str] = field(default_factory=list)
    fiducial_parameters: dict[str, Any] = field(default_factory=dict)
    prune_parameters: bool = False

    @classmethod
    def from_yaml(cls, path: str, mode: str | None = None) -> RunConfig:
        """Load a config yaml file and build a ``RunConfig`` object out of it.

        Parameters
        ----------
        path: str
            Path to the config yaml file.
        mode: str or None
            Run mode, see `from_dict`.

        Returns
        -------
        config: RunConfig
            The resolved ``RunConfig`` object.
        """
        config_path = Path(path).resolve()
        raw_text = config_path.read_text()
        return cls.from_dict(
            yaml.safe_load(raw_text), config_dir=config_path.parent,
            raw_text=raw_text, label=path, mode=mode,
        )

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], config_dir: Path, raw_text: str | None = None,
        label: str = "<dict>", mode: str | None = None,
    ) -> RunConfig:
        """Build a ``RunConfig`` object from an already parsed config.

        Parameters
        ----------
        raw: dict[str, Any]
            The parsed config, with the same layout as a config yaml file.
        config_dir: Path
            Directory relative paths in ``data`` are resolved against.
        raw_text: str or None
            Text saved next to the chain; ``raw`` dumped as yaml if ``None``.
        label: str
            Name of the config, used in the error messages.
        mode: str or None
            Run mode. With ``sdv``, ``sampler``, ``scale_cuts`` and
            ``data.covariance`` are optional, as they are not used.

        Returns
        -------
        config: RunConfig
            The resolved ``RunConfig`` object.
        """
        if raw_text is None:
            raw_text = yaml.safe_dump(raw, sort_keys=False)

        sdv = mode == "sdv"
        required = SDV_REQUIRED_SECTIONS if sdv else REQUIRED_SECTIONS
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(
                f"Config '{label}' is missing required section(s): {missing}"
            )

        if "kind" not in raw["probe"]:
            raise ValueError("'probe' section is missing required key 'kind'")
        sampler = raw.get("sampler", {})
        if "kind" not in sampler and not sdv:
            raise ValueError("'sampler' section is missing required key 'kind'")

        if raw["probe"]["kind"] in PHOTO_PROBES:
            data = PhotoDataConfig.from_dict(raw["data"], config_dir=config_dir, sdv=sdv)
            _check_photo_sampled_parameters(raw["sampled_parameters"])
            sampled_parameters = raw["sampled_parameters"]
            if "scale_cuts" in raw:
                scale_cuts = _parse_photo_scale_cuts(raw["scale_cuts"])
            elif sdv:
                scale_cuts = {}
            else:
                raise ValueError(
                    f"'scale_cuts' is required for probe '{raw['probe']['kind']}'"
                )
        else:
            data = DataConfig.from_dict(raw["data"], config_dir=config_dir)
            sampled_parameters = _normalize_sampled_parameters(
                raw["sampled_parameters"], data
            )
            scale_cuts = (
                _parse_scale_cuts(raw["scale_cuts"], data) if "scale_cuts" in raw else {}
            )
        derived_parameters = _auto_enable_derived_parameters(
            raw.get("derived_parameters", {}), raw["fixed_parameters"], sampled_parameters
        )

        return cls(
            probe=raw["probe"],
            data=data,
            scale_cuts=scale_cuts,
            sampled_parameters=sampled_parameters,
            fixed_parameters=raw["fixed_parameters"],
            sampler=sampler,
            output=raw["output"],
            raw_text=raw_text,
            constraints=raw.get("constraints", {}),
            derived_parameters=derived_parameters,
            derived_outputs=list(raw.get("derived_outputs", [])),
            fiducial_parameters=raw.get("fiducial_parameters", {}),
            prune_parameters=bool(raw.get("prune_parameters", False)),
        )
