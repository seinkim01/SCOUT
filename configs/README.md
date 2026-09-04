# configs/

Reference hyper-parameters for the experiments that ship with this repo.

These YAML files are **documentation**, not a config framework: each entry maps
one-to-one to a `--flag` of `src/core/train_linkpred.py` or
`src/core/train_nodeclf.py`. `argparse` remains the single source of truth; the
`scripts/run_*.sh` wrappers use the same values.
