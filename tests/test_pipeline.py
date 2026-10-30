from pathlib import Path

import numpy as np
import pytest
import yaml

from cloerun._core.config import RunConfig
from cloerun._core.pipeline import interpolate, parse_parameter, parse_parameters, to_run_config


@pytest.mark.parametrize(
    "value, fixed, prior, fiducial",
    [
        (0.06, 0.06, None, None),
        ("0.06", 0.06, None, None),
        ("40.0 67.0 91.0", None, {"dist": "flat", "min": 40.0, "max": 91.0}, 67.0),
        ([7.3, 7.8, 8.5], None, {"dist": "flat", "min": 7.3, "max": 8.5}, 7.8),
        ("gauss 0.96 0.042", None, {"dist": "norm", "loc": 0.96, "scale": 0.042}, 0.96),
        ("gauss 0.0, 0.01", None, {"dist": "norm", "loc": 0.0, "scale": 0.01}, 0.0),
        (
            {"dist": "flat", "min": 0.0, "max": 1.0, "fiducial": 0.5},
            None, {"dist": "flat", "min": 0.0, "max": 1.0}, 0.5,
        ),
    ],
)
def test_parse_parameter(value, fixed, prior, fiducial):
    assert parse_parameter("x", value) == (fixed, prior, fiducial)


@pytest.mark.parametrize(
    "value, match",
    [
        ("1.609 3.0443.664", "cannot read"),
        ("0.085 0.12057 0.085 0.2", "expected 'value'"),
        ("gauss 0.0", "gauss mean sigma"),
        ("1.0 0.5 2.0", "min <= start <= max"),
    ],
)
def test_parse_parameter_errors(value, match):
    with pytest.raises(ValueError, match=match):
        parse_parameter("x", value)


def test_parse_parameters_aliases_and_defaults():
    sections = parse_parameters({
        "cosmology": {"logAs": "1.6 3.0 3.6", "H0": 67.0, "Omega_k0": "-0.1 0.0 0.1"},
        "dz_she": {"dz_she_1": "gauss 0.0 0.01"},
    })
    assert sections["sampled_parameters"]["logA"]["dist"] == "flat"
    assert sections["fiducial_parameters"] == {"logA": 3.0, "Omega_k0": 0.0, "dz_shear_1": 0.0}
    assert "dz_shear_1" in sections["sampled_parameters"]
    # A sampled parameter is not also fixed to its default.
    assert "Omega_k0" not in sections["fixed_parameters"]
    assert sections["fixed_parameters"]["CIA"] == 0.0134
    assert sections["fixed_parameters"]["H0"] == 67.0


def test_parse_parameters_duplicate():
    with pytest.raises(ValueError, match="given twice"):
        parse_parameters({"a": {"H0": 67.0}, "b": {"H0": 68.0}})


def test_interpolate():
    value = {"a": ["${root_dir}/x.fits", 3], "b": "${name}"}
    assert interpolate(value, {"root_dir": "/data", "name": "run"}) == {
        "a": ["/data/x.fits", 3], "b": "run",
    }
    with pytest.raises(ValueError, match="Unknown variable"):
        interpolate("${nope}", {})


def _pipeline_config(probe="WL"):
    return {
        "run": {"mode": "chain"},
        "modelling": {
            "probe": probe,
            "parameters": {"cosmology": {"H0": "40.0 67.0 91.0", "w0": -1.0}},
            "nz": {"file": "nz_shear.txt", "zs_min": 1.0e-4, "zs_max": 3.0, "zs_num": 50},
            "theory": {
                "background": "CAMB",
                "linear_perturbations": "HMcode2020Emu",
                "nonlinear_perturbations": "HMcode2020Emu",
                "baryon_boost": None,
            },
            "systematics": {"ia_model": "NLA", "galaxy_bias_model": "per_bin"},
            "coupling": {"mode": "coupled", "mixing_matrices": "MM.fits"},
            "binning": {"kind": "log", "ell_min": 10, "ell_max": 1500, "N_ell": 8},
        },
        "inference": {
            "likelihood": {
                "data_vector": "cells.fits", "covariance": "cov.npz", "covariance_key": "Gauss",
            },
            "scale_cuts": {"kind": "per_block", "per_block": {"WL": [10, 1500], "GGL": [10, 750], "GC": [10, 750]}},
            "derived_outputs": ["sigma8_0", "S8"],
            "sampler": {"kind": "nautilus", "settings": {"n_live": 100}},
        },
    }


def test_to_run_config(tmp_path):
    raw, mode = to_run_config(_pipeline_config(), tmp_path, "run", str(tmp_path / "out"))
    assert mode == "chain"
    config = RunConfig.from_dict(raw, tmp_path)
    assert config.probe["kind"] == "WL"
    assert config.probe["linear_perturbations"] == "HMcode2020Emu"
    assert config.probe["settings"] == {"ia_model": "NLA", "galaxy_bias_model": "per_bin"}
    assert config.data.nz_shear == str(tmp_path / "nz_shear.txt")
    assert config.data.cells == str(tmp_path / "cells.fits")
    assert config.data.mixing == str(tmp_path / "MM.fits")
    assert config.data.z_grid == {"min": 1.0e-4, "max": 3.0, "num": 50}
    assert config.scale_cuts == {"WL": [10, 1500], "overrides": {}}
    assert config.derived_outputs == ["sigma8_0", "S8"]
    assert config.prune_parameters is True
    assert config.fiducial_parameters == {"H0": 67.0}
    assert config.output["sdv_file"] == str(tmp_path / "out" / "run_sdv.fits")


def test_to_run_config_sdv_and_decoupled(tmp_path):
    pipeline = _pipeline_config("3x2pt")
    pipeline["modelling"]["coupling"]["mode"] = "decoupled"
    raw, mode = to_run_config(pipeline, tmp_path, "run", str(tmp_path), mode="sdv")
    assert mode == "sdv"
    # The sdv mode writes the data vector, and decoupled runs need no mixing.
    assert "cells" not in raw["data"] and "mixing" not in raw["data"]
    # The 'inference' section is not read in the sdv mode.
    assert "covariance" not in raw["data"]
    assert "scale_cuts" not in raw and "sampler" not in raw
    config = RunConfig.from_dict(raw, tmp_path, mode="sdv")
    assert config.data.covariance is None
    assert config.scale_cuts == {} and config.sampler == {}


def test_to_run_config_sdv_without_inference(tmp_path):
    pipeline = _pipeline_config()
    pipeline["run"]["mode"] = "sdv"
    del pipeline["inference"]
    raw, mode = to_run_config(pipeline, tmp_path, "run", str(tmp_path))
    assert mode == "sdv"
    config = RunConfig.from_dict(raw, tmp_path, mode=mode)
    assert config.data.ells == {"kind": "log", "ell_min": 10, "ell_max": 1500, "N_ell": 8}
    # The other modes still need it.
    with pytest.raises(ValueError, match=r"missing section\(s\) \['inference'\]"):
        to_run_config(pipeline, tmp_path, "run", str(tmp_path), mode="chain")


def test_run_config_sdv_optional_sections(tmp_path):
    raw, _ = to_run_config(_pipeline_config(), tmp_path, "run", str(tmp_path))
    del raw["sampler"], raw["scale_cuts"], raw["data"]["covariance"]
    config = RunConfig.from_dict(raw, tmp_path, mode="sdv")
    assert config.data.covariance is None
    with pytest.raises(ValueError, match="missing required section"):
        RunConfig.from_dict(raw, tmp_path)
    raw["sampler"] = {"kind": "nautilus"}
    with pytest.raises(ValueError, match="missing required key 'covariance'"):
        RunConfig.from_dict(raw, tmp_path)
    raw["data"]["covariance"] = "cov.npz"
    with pytest.raises(ValueError, match="'scale_cuts' is required"):
        RunConfig.from_dict(raw, tmp_path)


def test_scale_cuts_from_file(tmp_path):
    (tmp_path / "cuts.yaml").write_text(yaml.safe_dump({
        "SHE-SHE-1-1": [40.0, 1500.0], "POS-SHE-1-1": [40.0, 750.0], "POS-POS-1-1": [40.0, 750.0],
    }))
    pipeline = _pipeline_config()
    pipeline["inference"]["scale_cuts"] = {"kind": "from_file", "from_file": "cuts.yaml"}
    raw, _ = to_run_config(pipeline, tmp_path, "run", str(tmp_path))
    config = RunConfig.from_dict(raw, tmp_path)
    # Only the WL spectra are kept, and the block has no default range.
    assert config.scale_cuts == {"WL": None, "overrides": {("SHE", "SHE", 1, 1): [40.0, 1500.0]}}


def test_to_run_config_rejects_unknown_probe(tmp_path):
    with pytest.raises(ValueError, match="modelling.probe"):
        to_run_config(_pipeline_config("Pk"), tmp_path, "run", str(tmp_path))


def test_photo_data_needs_cells_or_ells(tmp_path):
    raw, _ = to_run_config(_pipeline_config(), tmp_path, "run", str(tmp_path))
    del raw["data"]["cells"], raw["data"]["ells"]
    with pytest.raises(ValueError, match="needs 'cells', or 'ells'"):
        RunConfig.from_dict(raw, tmp_path)
