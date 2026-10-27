"""Records package versions for a given run.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import yaml

# Checked packages, always
TRACKED_PACKAGES = (
    "numpy", "scipy", "cloelib", "euclidlib", "cloelike", "cloerun",
)


def software_versions(extra_packages: Iterable[str] = ()) -> dict[str, str]:
    """Installed version of softwares.

    Parameters
    ----------
    extra_packages: Iterable[str]
        Specific package(s) that a cloerun run actually depends on (e.g.
        ``camb``, ``comet-emu``), on top of the packages always tracked
        in TRACKED_PACKAGES.

    Returns
    -------
    versions: dict[str, str]
        Installed version of each tracked package, or ``"version unknown"``
        if no installed-distribution metadata is found for it.
    """
    packages = list(TRACKED_PACKAGES) + [
        p for p in extra_packages if p not in TRACKED_PACKAGES
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "version unknown"
    return versions


def _strip_comment_lines(text: str) -> str:
    """Drop full-line YAML comments (lines whose only content is a ``#``
    comment).

    Parameters
    ----------
    text: str
        YAML source text.

    Returns
    -------
    stripped: str
        Same text, with comment-only lines removed.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    )


def write_config_with_versions(
    raw_text: str,
    directory: str | Path,
    name: str,
    extra_packages: Iterable[str] = (),
) -> Path:
    """Write config yaml with software version information.

    The result is still valid YAML, loadable by ``RunConfig.from_yaml``
    (which only reads the keys it knows about), so this file alone can be
    used to both reproduce the run and trace which package versions
    produced it.

    Parameters
    ----------
    raw_text: str
        The content of the input config yaml file. Comment-only lines are
        stripped before writing -- see `_strip_comment_lines`.
    directory: str or Path
        Directory to write the resulting file to (created if missing) --
        same convention as ``Chain.save``.
    name: str
        Base filename (the file is written as ``<name>.config.yaml``).
    extra_packages: Iterable[str]
        Forwarded to ``software_versions``.

    Returns
    -------
    path: Path
        Where the file was written.
    """
    version_block = yaml.safe_dump(
        {"software_versions": software_versions(extra_packages)}, sort_keys=False
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.config.yaml"
    stripped_text = _strip_comment_lines(raw_text).strip()
    path.write_text(f"{stripped_text}\n\n{version_block}")
    return path
