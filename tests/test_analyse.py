import numpy as np
import pytest

from cloerun._core.config import RunConfig
from cloerun.analyse import (
    analyse_run,
    default_plot_parameters,
    fiducial_values,
    summarise,
    summary_table,
)
from cloerun.chain import Chain


def _chain(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(1.0, 0.1, n)
    b = rng.normal(-2.0, 0.5, n)
    m1 = rng.normal(0.0, 0.01, n)
    samples = np.column_stack([np.full(n, -np.log(n)), np.zeros(n), a, b, m1, a + b])
    return Chain(["logweight", "loglikelihood", "a", "b", "m_1", "S8"], samples)


def test_summarise():
    summaries = {s.name: s for s in summarise(_chain(), {"a": 1.0, "b": -1.0})}
    assert summaries["a"].mean == pytest.approx(1.0, abs=0.005)
    assert summaries["a"].std == pytest.approx(0.1, rel=0.03)
    assert summaries["a"].low == pytest.approx(0.9, abs=0.01)
    assert summaries["a"].high == pytest.approx(1.1, abs=0.01)
    assert summaries["b"].shift == pytest.approx(-2.0, abs=0.05)
    assert summaries["S8"].fiducial is None and summaries["S8"].shift is None


def test_summary_weights():
    # Same values, but the second half carries all the weight.
    values = np.r_[np.zeros(100), np.ones(100)]
    logweight = np.r_[np.full(100, -np.inf), np.zeros(100)]
    chain = Chain(["logweight", "loglikelihood", "x"], np.column_stack([logweight, logweight, values]))
    (summary,) = summarise(chain)
    assert summary.mean == pytest.approx(1.0) and summary.std == pytest.approx(0.0, abs=1e-12)


def test_summary_table():
    table = summary_table(summarise(_chain(), {"a": 1.0}), title="run")
    assert table.startswith("# run\n")
    assert "| a | 1 |" in table
    # Header and one row per parameter.
    assert table.count("\n| ") == 5


def test_default_plot_parameters():
    assert default_plot_parameters(_chain()) == ["a", "b", "S8"]


def test_fiducial_values(tmp_path):
    config = RunConfig.from_dict(
        {
            "probe": {"kind": "GCspectro_BAO"},
            "data": {"redshifts": [1.0], "covariance": "x"},
            "fixed_parameters": {"c": 3.0, "m_{i}": 0.0},
            "sampled_parameters": {},
            "fiducial_parameters": {"a": 1.0, "unused_{i}": 2.0},
            "sampler": {"kind": "nautilus"},
            "output": {"directory": str(tmp_path), "name": "run"},
        },
        tmp_path,
    )
    assert fiducial_values(config, _chain()) == {"a": 1.0, "m_1": 0.0}


def test_analyse_run(tmp_path):
    pytest.importorskip("getdist")
    _chain().save(tmp_path, "run")
    config = RunConfig.from_dict(
        {
            "probe": {"kind": "GCspectro_BAO"},
            "data": {"redshifts": [1.0], "covariance": "x"},
            "fixed_parameters": {},
            "sampled_parameters": {},
            "fiducial_parameters": {"a": 1.0, "b": -2.0},
            "sampler": {"kind": "nautilus"},
            "output": {"directory": str(tmp_path), "name": "run"},
        },
        tmp_path,
    )
    summaries = analyse_run(config, parameters=["a", "b"])
    assert [s.name for s in summaries] == ["a", "b", "m_1", "S8"]
    assert (tmp_path / "run.summary.md").read_text().startswith("# run")
    assert (tmp_path / "run.triangle.pdf").stat().st_size > 0
    with pytest.raises(ValueError, match="Not in the chain"):
        analyse_run(config, parameters=["nope"])
