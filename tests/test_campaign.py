from pathlib import Path

import pytest
import yaml

from cloerun.campaign import apply_component, load_campaign, load_runs, main


def _config():
    return {
        "run": {"mode": "chain", "root_dir": "/data"},
        "modelling": {
            "probe": "WL",
            "parameters": "parameters.yaml",
            "nz": {"file": "${root_dir}/nz.fits", "zs_max": 3.0},
            "theory": {"background": "CAMB", "nonlinear_perturbations": "HMcode2020Emu"},
            "coupling": {"mode": "coupled", "mixing_matrices": "${root_dir}/MM.fits"},
            "binning": {"kind": "log", "ell_min": 10, "ell_max": 1500, "N_ell": 8},
        },
        "inference": {
            "likelihood": {"data_vector": "${root_dir}/cells.fits", "covariance": "cov.npz"},
            "scale_cuts": {"kind": "per_block", "per_block": {"WL": [10, 1500]}},
            "sampler": {
                "kind": "nautilus",
                "settings": {"n_live": 100, "filepath": "${output_dir}/${name}.hdf5"},
            },
        },
    }


def _write_campaign(tmp_path, runs, components):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(_config()))
    (tmp_path / "parameters.yaml").write_text(yaml.safe_dump(
        {"cosmology": {"H0": "40.0 67.0 91.0", "w0": -1.0, "wa": 0.0}}
    ))
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump({
        "output_dir": "out", "output_name": "campaign", "runs": runs, "components": components,
    }))
    return path


def test_campaign(tmp_path):
    path = _write_campaign(
        tmp_path,
        runs=[
            {"name": "baseline", "base": "config.yaml"},
            {"name": "w0wa", "parent": "baseline", "components": ["priors_w0wa", "contaminated"]},
            {"name": "w0wa_cuts", "parent": "w0wa", "components": ["cuts"]},
        ],
        components={
            "priors_w0wa": {"w0": "-3.0 -1.0 -0.33", "wa": "-3.0 0.0 3.0"},
            "contaminated": {"data_vector": "${root_dir}/contaminated.fits"},
            "cuts": {"scale_cuts": {"per_block": {"WL": [10, 300]}}},
        },
    )
    runs = {r.name: r for r in load_campaign(path)}
    assert list(runs) == ["baseline", "w0wa", "w0wa_cuts"]
    output_dir = str(tmp_path / "out" / "campaign")

    baseline = runs["baseline"]
    assert baseline.output_dir == output_dir
    assert baseline.config["modelling"]["parameters"]["cosmology"]["w0"] == -1.0
    assert baseline.config["inference"]["likelihood"]["data_vector"] == "/data/cells.fits"
    assert baseline.config["inference"]["sampler"]["settings"]["filepath"] == (
        f"{output_dir}/baseline.hdf5"
    )

    w0wa = runs["w0wa"]
    assert w0wa.config["modelling"]["parameters"]["cosmology"]["w0"] == "-3.0 -1.0 -0.33"
    assert w0wa.config["inference"]["likelihood"]["data_vector"] == "/data/contaminated.fits"
    # Unchanged keys of a merged section are kept.
    assert w0wa.config["inference"]["likelihood"]["covariance"] == "cov.npz"

    child = runs["w0wa_cuts"]
    assert child.config["inference"]["scale_cuts"] == {
        "kind": "per_block", "per_block": {"WL": [10, 300]},
    }
    assert child.config["modelling"]["parameters"]["cosmology"]["wa"] == "-3.0 0.0 3.0"
    config, mode = child.to_run_config()
    assert mode == "chain"
    assert config.output == {
        "directory": output_dir, "name": "w0wa_cuts",
        "sdv_file": f"{output_dir}/w0wa_cuts_sdv.fits",
    }
    assert config.sampled_parameters["w0"] == {"dist": "flat", "min": -3.0, "max": -0.33}


def test_variables_follow_components(tmp_path):
    path = _write_campaign(
        tmp_path,
        runs=[
            {"name": "baseline", "base": "config.yaml"},
            {"name": "moved", "parent": "baseline", "components": ["moved"]},
        ],
        components={"moved": {"run": {"root_dir": "/elsewhere"}}},
    )
    runs = {r.name: r for r in load_campaign(path)}
    assert runs["moved"].config["modelling"]["nz"]["file"] == "/elsewhere/nz.fits"


def test_dotted_and_ambiguous_keys():
    config = _config()
    with pytest.raises(ValueError, match="ambiguous"):
        apply_component(config, {"mode": "decoupled"}, "c")
    updated = apply_component(config, {"modelling.coupling.mode": "decoupled"}, "c")
    assert updated["modelling"]["coupling"]["mode"] == "decoupled"
    assert updated["run"]["mode"] == "chain"
    assert config["modelling"]["coupling"]["mode"] == "coupled"
    with pytest.raises(ValueError, match="not in the config"):
        apply_component(config, {"nope": 1}, "c")


@pytest.mark.parametrize(
    "runs, components, match",
    [
        ([{"name": "a", "parent": "b"}, {"name": "b", "parent": "a"}], {}, "their own parent"),
        ([{"name": "a", "parent": "missing"}], {}, "Unknown run"),
        ([{"name": "a", "base": "config.yaml", "components": ["x"]}], {}, "unknown component"),
        ([{"name": "a"}], {}, "exactly one of"),
        ([{"name": "a", "base": "config.yaml"}] * 2, {}, "defined twice"),
    ],
)
def test_campaign_errors(tmp_path, runs, components, match):
    with pytest.raises(ValueError, match=match):
        load_campaign(_write_campaign(tmp_path, runs, components))


def test_single_pipeline_config(tmp_path):
    _write_campaign(tmp_path, runs=[], components={})
    (run,) = load_campaign(tmp_path / "config.yaml")
    assert run.name == "config"
    assert run.output_dir == str(tmp_path / "chains")


def test_cli_dry_run(tmp_path, capsys):
    path = _write_campaign(
        tmp_path,
        runs=[{"name": "a", "base": "config.yaml"}, {"name": "b", "base": "config.yaml"}],
        components={},
    )
    main([str(path), "--run", "b", "--mode", "sdv", "--dry-run"])
    out = capsys.readouterr().out
    assert "=== b (sdv)" in out and "=== a" not in out
    with pytest.raises(ValueError, match="Unknown run"):
        main([str(path), "--run", "c", "--dry-run"])


def test_cloerun_dispatcher(tmp_path, capsys):
    from cloerun.__main__ import main as cloerun

    with pytest.raises(SystemExit) as exit_info:
        cloerun([])
    assert exit_info.value.code == 2
    assert "commands:" in capsys.readouterr().out
    path = _write_campaign(tmp_path, runs=[{"name": "a", "base": "config.yaml"}], components={})
    cloerun(["campaign", str(path), "--dry-run"])
    assert "=== a (chain)" in capsys.readouterr().out


def test_load_runs_native_config():
    from cloerun.campaign import load_runs

    examples = Path(__file__).parents[1] / "examples"
    ((name, config, mode),) = load_runs(examples / "wl_nautilus.yaml")
    assert (name, mode) == ("test_wl_NLA_HMcode", "chain")
    assert config.probe["kind"] == "WL"


def test_cloerun_config_as_base(tmp_path):
    examples = Path(__file__).parents[1] / "examples"
    (tmp_path / "bao.yaml").write_text((examples / "gcspectro_bao_nautilus.yaml").read_text())
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump({
        "output_dir": "out", "output_name": "spectro",
        "runs": [
            {"name": "bao", "base": "bao.yaml", "components": ["checkpoint"]},
            {"name": "bao_bbn", "parent": "bao", "components": ["bbn"]},
        ],
        "components": {
            "checkpoint": {"sampler.settings": {"filepath": "${output_dir}/${name}.hdf5"}},
            "bbn": {"ombh2": {"dist": "norm", "loc": 0.022, "scale": 0.0001}},
        },
    }))
    runs = {name: (config, mode) for name, config, mode in load_runs(path)}
    config, mode = runs["bao_bbn"]
    output_dir = str(tmp_path / "out" / "spectro")
    assert mode == "chain"
    assert config.probe["kind"] == "GCspectro_BAO"
    assert config.output == {"directory": output_dir, "name": "bao_bbn"}
    assert config.sampler["settings"]["filepath"] == f"{output_dir}/bao_bbn.hdf5"
    assert config.sampled_parameters["ombh2"]["scale"] == 0.0001
    assert runs["bao"][0].sampled_parameters["ombh2"]["scale"] == 0.00038
    # Data paths stay relative to the base config.
    assert config.data.covariance_template == str(tmp_path / "../data/EUC_LE3_BAO_COV_Z{}.fits")
