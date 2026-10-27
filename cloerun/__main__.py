"""``cloerun <command> FILE``, also ``python -m cloerun <command> FILE``.

Commands: ``campaign`` (each run in its ``run.mode``), ``sdv``, ``chain`` and
``analyse``; ``cloerun <command> --help`` lists their options.
"""

from __future__ import annotations

import sys

COMMANDS = {
    "campaign": "Run each run in its 'run.mode' (chain, sdv or test).",
    "sdv": "Write the theory at the fiducial point as a synthetic data vector.",
    "chain": "Sample the posterior (--test: one loglike at the fiducial point).",
    "analyse": "Summary table and triangle plot of finished chains.",
}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        usage = "\n".join(f"  {name:<9} {help}" for name, help in COMMANDS.items())
        print(f"usage: cloerun <command> FILE [options]\n\ncommands:\n{usage}")
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    command, *rest = argv
    # Imported here, so that `cloerun --help` does not load the theory codes.
    if command == "campaign":
        from cloerun.campaign import main as run
    elif command == "sdv":
        from cloerun.sdv import main as run
    elif command == "chain":
        from cloerun.chain import main as run
    else:
        from cloerun.analyse import main as run
    run(rest)


if __name__ == "__main__":
    main()
