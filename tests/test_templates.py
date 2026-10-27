import pytest

from cloerun._core.templates import expand_bin_templates

NAMES = [
    "H0",
    "AIA",
    *[f"dz_shear_{i}" for i in range(1, 4)],
    *[f"b1_photo_bin{i}" for i in range(3)],
    *[f"b1_photo_nl_bin{i}" for i in range(3)],
]


def test_expands_with_the_index_convention_of_each_family():
    expanded = expand_bin_templates(
        {"dz_shear_{i}": 0.0, "b1_photo_bin{i}": 1.0}, NAMES, "fixed_parameters"
    )
    assert expanded == {
        "dz_shear_1": 0.0,
        "dz_shear_2": 0.0,
        "dz_shear_3": 0.0,
        "b1_photo_bin0": 1.0,
        "b1_photo_bin1": 1.0,
        "b1_photo_bin2": 1.0,
    }


def test_template_does_not_match_a_longer_prefix():
    expanded = expand_bin_templates({"b1_photo_bin{i}": 1.0}, NAMES, "fixed_parameters")
    assert not any("nl" in name for name in expanded)


def test_explicit_entry_overrides_template_regardless_of_order():
    expanded = expand_bin_templates(
        {"dz_shear_2": 0.5, "dz_shear_{i}": 0.0}, NAMES, "fixed_parameters"
    )
    assert expanded["dz_shear_2"] == 0.5
    assert expanded["dz_shear_1"] == expanded["dz_shear_3"] == 0.0


def test_entries_without_placeholder_are_kept():
    expanded = expand_bin_templates({"H0": 70.0, "unused": 1.0}, NAMES, "fixed_parameters")
    assert expanded == {"H0": 70.0, "unused": 1.0}


def test_unmatched_template_raises():
    with pytest.raises(ValueError, match="dz_shaer"):
        expand_bin_templates({"dz_shaer_{i}": 0.0}, NAMES, "sampled_parameters")


def test_prune_unused():
    from cloerun._core.templates import prune_unused

    section = {"dz_shear_{i}": 0.0, "dz_pos_{i}": 0.0, "b1_photo_bin0": 1.0, "omch2": 0.12, "H0": 67.0}
    pruned, dropped = prune_unused(section, ["dz_shear_1", "H0"], keep=["omch2"])
    assert pruned == {"dz_shear_{i}": 0.0, "omch2": 0.12, "H0": 67.0}
    assert dropped == ["dz_pos_{i}", "b1_photo_bin0"]
