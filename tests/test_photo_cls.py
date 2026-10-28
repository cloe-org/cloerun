import numpy as np
import pytest

pytest.importorskip("cloelike")

from cloerun._core.probes.photo_cls import _block_sizes, _select_covariance  # noqa: E402


def _labelled_covariance(sizes):
    """Diagonal covariance whose entries tell which block they belong to."""
    labels = {"WL": 1.0, "GGL": 2.0, "GCph": 3.0}
    return np.diag(
        np.concatenate([np.full(sizes[b], labels[b]) for b in ("WL", "GGL", "GCph")])
    )


def test_block_sizes():
    assert _block_sizes(n_pos=6, n_she=6, n_ell=32) == {
        "WL": 21 * 32,
        "GGL": 36 * 32,
        "GCph": 21 * 32,
    }


@pytest.mark.parametrize(
    "kind, blocks",
    [
        ("3x2pt", [1.0, 2.0, 3.0]),
        ("WL", [1.0]),
        ("GGL", [2.0]),
        ("GCph", [3.0]),
        ("2x2pt", [2.0, 3.0]),
    ],
)
def test_select_covariance_from_full_3x2pt(kind, blocks):
    sizes = _block_sizes(n_pos=3, n_she=2, n_ell=4)
    selected = _select_covariance(_labelled_covariance(sizes), kind, sizes)
    assert list(dict.fromkeys(np.diag(selected))) == blocks


def test_select_covariance_of_the_probe_only():
    sizes = _block_sizes(n_pos=3, n_she=2, n_ell=4)
    covariance = np.eye(sizes["WL"])
    assert _select_covariance(covariance, "WL", sizes) is covariance


def test_select_covariance_wrong_size():
    sizes = _block_sizes(n_pos=3, n_she=2, n_ell=4)
    with pytest.raises(ValueError, match="matches neither"):
        _select_covariance(np.eye(5), "WL", sizes)


def test_binning_ells(tmp_path):
    from cloerun._core.probes.photo_cls import binning_ells

    ells = binning_ells({"kind": "log", "ell_min": 10, "ell_max": 1000, "N_ell": 2})
    np.testing.assert_allclose(ells, [55.0, 550.0])
    ells = binning_ells({"kind": "linear", "ell_min": 0, "ell_max": 10, "N_ell": 5})
    np.testing.assert_allclose(ells, [1, 3, 5, 7, 9])
    path = tmp_path / "ell_values.txt"
    path.write_text("# ell delta_ell\n10.5 2.0\n12.5 2.0\n")
    np.testing.assert_allclose(binning_ells({"kind": "fromfile", "file": str(path)}), [10.5, 12.5])
    with pytest.raises(ValueError, match="Unknown 'ells.kind'"):
        binning_ells({"kind": "cubic"})


def test_load_nz_text(tmp_path):
    from cloerun._core.probes.photo_cls import _load_nz

    z = np.linspace(0.0, 2.0, 201)
    table = np.column_stack([z, np.exp(-((z - 0.5) / 0.1) ** 2), 3.0 * np.exp(-((z - 1.0) / 0.2) ** 2)])
    path = tmp_path / "nz_shear.txt"
    np.savetxt(path, table, header="z bin1 bin2")
    nz = _load_nz(str(path), z)
    assert nz.shape == (2, z.size)
    np.testing.assert_allclose(np.trapezoid(nz, z, axis=1), 1.0)


def test_template_cells():
    from cloerun._core.probes.photo_cls import _template_cells

    cells = _template_cells("3x2pt", n_pos=2, n_she=3, n_ell=4)
    assert sum(k[:2] == ("SHE", "SHE") for k in cells) == 6
    assert sum(k[:2] == ("POS", "SHE") for k in cells) == 6
    assert sum(k[:2] == ("POS", "POS") for k in cells) == 3
    assert cells[("SHE", "SHE", 1, 3)].shape == (1, 1, 4)
    assert cells[("POS", "SHE", 2, 1)].shape == (1, 4)
    assert cells[("POS", "POS", 1, 2)].shape == (4,)
