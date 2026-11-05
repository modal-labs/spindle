"""Run full-model replay checks in fresh processes on an isolated Hopper trainer.

PYTHONPATH=src modal run scripts/check_deterministic_training_gpu.py
"""

import json
import os
from pathlib import Path

import modal

from determinism_report import compare
from spindle.providers.modal.image_dependencies import ignore_config_source
from spindle.providers.modal.miles_image import runtime_image

ROOT = Path(__file__).resolve().parents[1]
TEST_TP = int(os.environ.get("SPINDLE_DETERMINISM_TP", "1"))
if TEST_TP not in (1, 2, 4, 8):
    raise ValueError("SPINDLE_DETERMINISM_TP must be 1, 2, 4, or 8")
app = modal.App("spindle-deterministic-training-validation")
base_image = runtime_image
if os.environ.get("SPINDLE_MILES_FA3_ARTIFACTS"):
    from spindle.providers.modal.miles_fa3_image import with_deterministic_fa3

    base_image = with_deterministic_fa3(
        base_image, Path(os.environ["SPINDLE_MILES_FA3_ARTIFACTS"])
    )
probe_image = (
    base_image.env(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NCCL_ALGO": "Ring",
            "NCCL_PROTO": "Simple",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "SPINDLE_DETERMINISM_TP": str(TEST_TP),
        }
    )
    .add_local_python_source("spindle", ignore=ignore_config_source)
    .add_local_file(
        ROOT / "tests/manual/deterministic_training_probe.py",
        "/root/deterministic_training_probe.py",
    )
    .add_local_file(
        ROOT / "scripts/determinism_report.py", "/root/determinism_report.py"
    )
    .add_local_file(
        ROOT / "tests/manual/deterministic_probe_actor.py",
        "/root/deterministic_probe_actor.py",
    )
)


@app.function(
    image=probe_image,
    gpu=f"H100!:{TEST_TP}",
    cpu=16,
    memory=98304 * TEST_TP,
    timeout=3600,
    max_containers=1,
)
def probe(model, steps=12, deterministic=True):
    import os
    import subprocess
    import sys
    import tempfile
    from huggingface_hub import snapshot_download

    root = Path(tempfile.mkdtemp(prefix="deterministic-training-"))
    checkpoint = snapshot_download(model, ignore_patterns=["*.bin", "*.pt"])
    reports = {}
    result = {
        "model": model,
        "steps": steps,
        "deterministic": deterministic,
        "tp": TEST_TP,
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip(),
        "reports": reports,
    }
    for mode in ("reference", "repeat", "packed", "restore"):
        output = root / f"{mode}.json"
        command = [
            sys.executable,
            "/root/deterministic_training_probe.py",
            "--model",
            model,
            "--checkpoint",
            checkpoint,
            "--checkpoint-state",
            str(root / "reference/checkpoints/initial/target"),
            "--reference",
            str(root / "reference.json"),
            "--mode",
            mode,
            "--steps",
            str(steps),
            "--tp",
            str(TEST_TP),
            "--output",
            str(output),
        ]
        if deterministic:
            command.append("--deterministic")
        proc = subprocess.run(
            command, cwd=root, env={**os.environ, "PYTHONPATH": "/root", "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0" if deterministic else "1"}, timeout=800
        )
        if output.exists():
            reports[mode] = json.loads(output.read_text())
        if proc.returncode:
            result.update(passed=False, error=f"{mode} exited {proc.returncode}")
            return result
    result["comparisons"] = {
        mode: compare(reports["reference"], reports[mode])
        for mode in ("repeat", "packed", "restore")
    }
    if "same_process" in reports["reference"]:
        result["comparisons"]["same_process"] = compare(
            reports["reference"], reports["reference"]["same_process"]
        )
    result["passed"] = all(row["equal"] for row in result["comparisons"].values())
    return result


@app.local_entrypoint()
def main(
    models: str = "Qwen/Qwen3-0.6B,Qwen/Qwen3-8B",
    steps: int = 12,
    deterministic: bool = True,
    output: str = "scripts/results/deterministic-training/report.json",
):
    if steps < 8:
        raise ValueError(
            "Use at least eight steps to cover the reported step-3/step-6 divergence"
        )
    reports = []
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for model in models.split(","):
        reports.append(probe.remote(model.strip(), steps, deterministic))
        path.write_text(json.dumps(reports, indent=2) + "\n")
        print(
            json.dumps(
                {key: value for key, value in reports[-1].items() if key != "reports"},
                indent=2,
            )
        )
    if not all(report["passed"] for report in reports):
        raise SystemExit(1)
