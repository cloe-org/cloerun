"""cloerun: run Euclid inference campaigns with cloelike likelihoods.

The interface is four modules, one per step: `cloerun.campaign` (resolve and
run campaigns), `cloerun.sdv` (synthetic data vectors), `cloerun.chain`
(sample the posterior) and `cloerun.analyse` (summaries and plots).
"""

from cloerun._core.config import RunConfig
from cloerun.campaign import load_runs, run_campaign
from cloerun.chain import Chain, run_chain, test_loglike
from cloerun.sdv import write_sdv

__all__ = [
    "Chain", "RunConfig", "load_runs", "run_campaign", "run_chain", "test_loglike", "write_sdv",
]
