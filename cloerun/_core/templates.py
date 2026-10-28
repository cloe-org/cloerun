"""Expansion of per-bin parameter name templates.

Photometric likelihoods take one scalar parameter per tomographic bin (e.g.
``dz_shear_1``, ..., ``dz_shear_N`` or ``b1_photo_bin0``, ...). Writing them
all out in the config yaml is tedious and error-prone, so a parameter name in
``fixed_parameters`` or ``sampled_parameters`` may contain the placeholder
``{i}`` (e.g. ``dz_shear_{i}``), which is expanded to every parameter name the
likelihood actually uses that matches it -- whatever the index convention of
that family (1-based for ``dz_shear_{i}``, 0-based for ``b1_photo_bin{i}``).
"""

from __future__ import annotations

import re
from typing import Any, Iterable

PLACEHOLDER = "{i}"


def _template_regex(template: str) -> re.Pattern:
    """Turn a name template into a regex matching its expansions.

    Parameters
    ----------
    template: str
        Parameter name containing the ``{i}`` placeholder.

    Returns
    -------
    regex: re.Pattern
        Compiled regex, capturing the bin index of a matching name.
    """
    prefix, _, suffix = template.partition(PLACEHOLDER)
    return re.compile(f"{re.escape(prefix)}(\\d+){re.escape(suffix)}")


def expand_bin_templates(
    section: dict[str, Any], known_names: Iterable[str], label: str
) -> dict[str, Any]:
    """Expand the ``{i}`` templates of a config section into one entry per bin.

    Entries without a placeholder are kept as they are and take precedence
    over a template expanding to the same name, so a template can set a
    default that single bins override.

    Parameters
    ----------
    section: dict[str, Any]
        The ``fixed_parameters`` or ``sampled_parameters`` section.
    known_names: Iterable[str]
        Every parameter name the likelihood consumes.
    label: str
        Name of the section, used in the error message.

    Returns
    -------
    expanded: dict[str, Any]
        The section with each template replaced by its expansions, ordered by
        bin index, at the position of the template.
    """
    known_names = list(known_names)
    explicit = {name for name in section if PLACEHOLDER not in name}
    expanded: dict[str, Any] = {}
    for name, spec in section.items():
        if PLACEHOLDER not in name:
            expanded[name] = spec
            continue
        regex = _template_regex(name)
        matches = sorted(
            (int(m.group(1)), candidate)
            for candidate in known_names
            if (m := regex.fullmatch(candidate))
        )
        if not matches:
            raise ValueError(
                f"'{label}.{name}' does not match any parameter of the likelihood"
            )
        for _, candidate in matches:
            if candidate not in explicit:
                expanded[candidate] = spec
    return expanded


def prune_unused(
    section: dict[str, Any], known_names: Iterable[str], keep: Iterable[str] = ()
) -> tuple[dict[str, Any], list[str]]:
    """Drop the entries of a config section that the likelihood does not
    read, for parameter files shared by several probes (e.g. the galaxy bias
    of a WL run).

    Parameters
    ----------
    section: dict[str, Any]
        The ``fixed_parameters`` or ``sampled_parameters`` section.
    known_names: Iterable[str]
        Every parameter name the likelihood consumes.
    keep: Iterable[str]
        Names to keep anyway, e.g. the inputs of derived-parameter relations
        (``omch2``, ``logA``, ...).

    Returns
    -------
    pruned: dict[str, Any]
        The section without the unused entries.
    dropped: list[str]
        Names of the dropped entries.
    """
    known_names = list(known_names)
    keep = set(keep)

    def used(name: str) -> bool:
        if PLACEHOLDER not in name:
            return name in keep or name in known_names
        regex = _template_regex(name)
        return any(regex.fullmatch(candidate) for candidate in known_names)

    pruned = {name: spec for name, spec in section.items() if used(name)}
    return pruned, [name for name in section if name not in pruned]
