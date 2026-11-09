import numpy as np
import pytest

pytest.importorskip("cloelike")

from cloerun._core.probes.photo_2pcf import (  # noqa: E402
    _block_sizes,
    _template_twopcf,
    _theta_range,
    binning_theta,
    read_correlation_functions,
    split_vector,
    write_spectra,
)


def test_block_sizes():
    assert _block_sizes(n_pos=6, n_she=6, n_theta=20) == {
        "xi_plus": 21 * 20, "xi_minus": 21 * 20, "gamma_t": 36 * 20, "w": 21 * 20,
    }


def test_binning_theta():
    theta = binning_theta({"theta_min": 1.0, "theta_max": 100.0, "N_theta": 2})
    np.testing.assert_allclose(theta, [np.sqrt(10.0), np.sqrt(1000.0)])


def test_theta_range():
    assert _theta_range([2.5, 250, 10, 250], "WL", "SHE-SHE-1-1") == [2.5, 250.0, 10.0, 250.0]
    assert _theta_range({"plus": [2.5, 250], "minus": [10, 250]}, "WL", "x") == [2.5, 250.0, 10.0, 250.0]
    assert _theta_range([6, 250], "GGL", "x") == [6.0, 250.0]
    with pytest.raises(ValueError, match="expected 4 numbers"):
        _theta_range([2.5, 250], "WL", "x")
    with pytest.raises(ValueError, match="expected 2 numbers"):
        _theta_range([2.5, 250, 10, 250], "GCph", "x")


def test_template_and_split_vector():
    n_pos, n_she, n_theta = 2, 3, 4
    template = _template_twopcf("3x2pt_2pcf", n_pos, n_she, n_theta)
    assert template[("SHE", "SHE", 1, 3)].shape == (2, 2, n_theta)
    assert template[("POS", "SHE", 2, 1)].shape == (2, n_theta)
    assert template[("POS", "POS", 1, 2)].shape == (n_theta,)
    sizes = _block_sizes(n_pos, n_she, n_theta)
    vector = np.arange(sum(sizes.values()), dtype=float)
    twopcf = split_vector(vector, "3x2pt_2pcf", n_pos, n_she, n_theta)
    np.testing.assert_allclose(twopcf[("SHE", "SHE", 1, 1)][0, 0], vector[:n_theta])
    np.testing.assert_allclose(twopcf[("SHE", "SHE", 1, 1)][1, 1], vector[sizes["xi_plus"]:sizes["xi_plus"] + n_theta])
    start = sizes["xi_plus"] + sizes["xi_minus"]
    np.testing.assert_allclose(twopcf[("POS", "SHE", 1, 1)][0], vector[start:start + n_theta])
    np.testing.assert_allclose(twopcf[("POS", "POS", 2, 2)], vector[-n_theta:])
    with pytest.raises(ValueError, match="does not match"):
        split_vector(vector[:-1], "3x2pt_2pcf", n_pos, n_she, n_theta)


def test_write_read_correlation_functions(tmp_path):
    pytest.importorskip("fitsio")
    n_pos, n_she, n_theta = 2, 2, 5
    theta = np.geomspace(2.5, 250.0, n_theta)
    rng = np.random.default_rng(0)
    twopcf = _template_twopcf("3x2pt_2pcf", n_pos, n_she, n_theta)
    for key, value in twopcf.items():
        if key[:2] == ("SHE", "SHE"):
            value[0, 0] = rng.normal(size=n_theta)
            value[1, 1] = rng.normal(size=n_theta)
        elif key[:2] == ("POS", "SHE"):
            value[0] = rng.normal(size=n_theta)
        else:
            value[:] = rng.normal(size=n_theta)
    path = tmp_path / "twopcf.fits"
    write_spectra(str(path), {"theta": theta, "twopcf": twopcf})
    theta_read, read = read_correlation_functions(str(path))
    np.testing.assert_allclose(theta_read, theta)
    assert set(read) == set(twopcf)
    for key in twopcf:
        np.testing.assert_allclose(read[key], twopcf[key])
