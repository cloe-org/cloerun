"""Data model for the sampled posterior distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cloerun._core.priors import Priors

# Columns that always come before the sampled parameters.
RESERVED_COLUMNS = ["logweight", "loglikelihood"]


@dataclass
class Chain:
    """Sampled posterior distribution.

    Columns are always in this format:
    ``[logweight, loglikelihood, *params, *derived]``.

    Attributes
    ----------
    columns: list[str]
        Name of each column in ``samples``, in order.
    samples: np.ndarray
        2D array of shape ``(n_samples, len(columns))``.
    """

    columns: list[str]
    samples: np.ndarray

    @classmethod
    def build(cls, priors: Priors, points: np.ndarray,
              logweight: np.ndarray, loglikelihood: np.ndarray,
              derived: np.ndarray | None = None,
              derived_names: list[str] = ()) -> Chain:
        """Build a `Chain` object from a sampler raw output.

        Parameters
        ----------
        priors: Priors
            Sampled parameters and their distributions.
        points: np.ndarray
            Samples, with columns in ``priors.names`` order.
        logweight: np.ndarray
            Log-weight of each sample, as returned by the sampler.
        loglikelihood: np.ndarray
            Log-likelihood of each sample, as returned by the sampler.
        derived: np.ndarray or None
            Derived quantities of each sample, of shape
            ``(n_samples, len(derived_names))``.
        derived_names: list[str]
            Names of the derived quantities.

        Returns
        -------
        chain: Chain
            The standardized `Chain` object, with columns in the order
            ``[logweight, loglikelihood, *priors.names, *derived_names]``.
        """
        columns = RESERVED_COLUMNS + priors.names + list(derived_names)
        blocks = [logweight, loglikelihood, points]
        if derived is not None:
            blocks.append(derived)
        samples = np.column_stack(blocks)
        return cls(columns=columns, samples=samples)

    def save(self, path: str | Path, filename: str) -> Path:
        """Save the chain to a compressed binary file and return the full path.

        Parameters
        ----------
        path: str or Path
            Path to write the file to (created if missing).
        filename: str
            Root filename (the file is written as ``<filename>.npz``).

        Returns
        -------
        full_path: Path
            Full path to the saved file.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        full_path = directory / f"{filename}.npz"
        np.savez_compressed(full_path, samples=self.samples, columns=np.array(self.columns))
        return full_path

    @classmethod
    def load(cls, path: str | Path) -> Chain:
        """Load a chain previously written by `Chain.save` and store it into a `Chain` object.

        Parameters
        ----------
        path: str or Path
            Path to the binary file.

        Returns
        -------
        chain: Chain
        """
        data = np.load(path)
        return cls(columns=list(data["columns"]), samples=data["samples"])

    @property
    def logweight(self) -> np.ndarray:
        """Log-weight of each sample."""
        return self.column("logweight")
    
    @property
    def loglikelihood(self) -> np.ndarray:
        """Log-likelihood of each sample."""
        return self.column("loglikelihood")
    
    @property
    def parameters(self) -> list[str]:
        """Sampled parameter and derived quantity names, excluding the
        reserved columns."""
        return self.columns[len(RESERVED_COLUMNS):]

    def column(self, name: str) -> np.ndarray:
        """Values of one column by name.

        Parameters
        ----------
        name: str
            Column name.

        Returns
        -------
        values: np.ndarray
        """
        return self.samples[:, self.columns.index(name)]
