"""Minimal test: does forking after JAX has started deadlock on this machine?"""

import multiprocessing
import sys
import time

import jax.numpy as jnp


def work(i):
    return float(jnp.sum(jnp.arange(1000.0) * i))


if __name__ == "__main__":
    method = sys.argv[1]  # "fork" or "spawn"
    print("result in the main process:", work(1))  # starts JAX before the workers
    start = time.time()
    with multiprocessing.get_context(method).Pool(4) as pool:
        results = pool.map(work, range(8))
    print(f"{method}: {len(results)} results in {time.time() - start:.1f} s")
