from pathlib import Path

import pytest
import yaml

from cloerun._core.config import DataConfig, PhotoDataConfig, RunConfig

EXAMPLES = Path(__file__).parents[1] / "examples"


def _write(tmp_path, raw):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _photo_raw():
    return yaml.safe_load((EXAMPLES / "3x2pt_nautilus.yaml").read_text())


@pytest.mark.parametrize(
    "example, data_type",
    [
        ("3x2pt_nautilus.yaml", PhotoDataConfig),
        ("wl_nautilus.yaml", PhotoDataConfig),
        ("gcspectro_pls_nautilus.yaml", DataConfig),
        ("gcspectro_bao_nautilus.yaml", DataConfig),
        ("gcspectro_pls_bao_nautilus.yaml", DataConfig),
    ],
)
def test_examples_parse(example, data_type):
    config = RunConfig.from_yaml(EXAMPLES / example)
    assert isinstance(config.data, data_type)


def test_photo_example():
    config = RunConfig.from_yaml(EXAMPLES / "3x2pt_nautilus.yaml")
    assert config.data.nz_positions == config.data.nz_shear
    assert config.data.noise_free_mock is True
    assert config.scale_cuts["WL"] == [10, 1500]
    assert config.scale_cuts["overrides"] == {}
    # omch2, ombh2 and logA trigger the conversions to the native parameters.
    assert config.derived_parameters == {
        "Omega_cdm0_from_omch2": True,
        "Omega_b0_from_ombh2": True,
        "As_from_logA": True,
    }


def test_photo_scale_cut_overrides(tmp_path):
    raw = _photo_raw()
    raw["scale_cuts"]["overrides"] = {"POS-SHE-1-3": [10, 300]}
    config = RunConfig.from_yaml(_write(tmp_path, raw))
    assert config.scale_cuts["overrides"] == {("POS", "SHE", 1, 3): [10, 300]}


@pytest.mark.parametrize(
    "scale_cuts, match",
    [
        ({"WL": [10, 100], "CMB": [1, 2]}, "Unexpected"),
        ({"WL": [10, 100], "overrides": {"SHE-POS-1-1": [1, 2]}}, "Invalid"),
        ({"WL": [10, 100], "overrides": {"SHE-SHE-1": [1, 2]}}, "Invalid"),
    ],
)
def test_photo_scale_cuts_errors(tmp_path, scale_cuts, match):
    raw = _photo_raw()
    raw["scale_cuts"] = scale_cuts
    with pytest.raises(ValueError, match=match):
        RunConfig.from_yaml(_write(tmp_path, raw))


def test_photo_separate_nz(tmp_path):
    raw = _photo_raw()
    del raw["data"]["nz"]
    raw["data"]["nz_positions"] = "lens.fits"
    raw["data"]["nz_shear"] = "source.fits"
    config = RunConfig.from_yaml(_write(tmp_path, raw))
    assert config.data.nz_positions.endswith("lens.fits")
    assert config.data.nz_shear.endswith("source.fits")


def test_photo_missing_nz(tmp_path):
    raw = _photo_raw()
    del raw["data"]["nz"]
    with pytest.raises(ValueError, match="nz"):
        RunConfig.from_yaml(_write(tmp_path, raw))


@pytest.mark.parametrize(
    "spec, match",
    [
        ({"dist": "AM", "loc": 0.0, "scale": 1.0}, "analytical marginalisation"),
        ({1.0: {"dist": "flat", "min": 0, "max": 1}}, "single spec"),
    ],
)
def test_photo_rejects_spectro_only_specs(tmp_path, spec, match):
    raw = _photo_raw()
    raw["sampled_parameters"]["AIA"] = spec
    with pytest.raises(ValueError, match=match):
        RunConfig.from_yaml(_write(tmp_path, raw))


def test_spectro_num_mocks(tmp_path):
    raw = yaml.safe_load((EXAMPLES / "gcspectro_bao_nautilus.yaml").read_text())
    assert RunConfig.from_yaml(_write(tmp_path, raw)).data.num_mocks is None
    raw["data"]["num_mocks"] = 1000
    assert RunConfig.from_yaml(_write(tmp_path, raw)).data.num_mocks == 1000
