# Spindle benchmarks

Workloads and configs live here. The runner and local dashboard ship with Spindle.
See [the benchmark guide](../docs/benchmarking.md) for setup, telemetry, result
uploads, and comparison rules. No shared dashboard or credentials are required.

Run `python -m spindle.benchmarks validate benchmarks/configs/trainer.json` before
starting a GPU run. Copy a config to add a workload or tune a deployment.
