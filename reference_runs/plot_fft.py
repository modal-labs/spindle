"""Plot recorded FFT client latency, throughput, and GPU cost.

uv run python reference_runs/plot_fft.py reference_runs/results/qwen35-4b-fft-blackwell/measurements.json
"""

import argparse
import json
from pathlib import Path

from matplotlib import pyplot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    args = parser.parse_args()
    data = json.loads(args.measurements.read_text())
    contexts = sorted({p["context_length"] for r in data["runs"] for p in r["points"]})
    layouts = sorted(
        {(r["hardware"], r["tp"], r["cp"]) for r in data["runs"]}, reverse=True
    )
    plots = [
        (
            "latency-throughput",
            "Median client step latency (s)",
            "Input throughput (k tokens/s)",
        ),
        (
            "step-latency",
            "Sequences per client update",
            "Median client step latency (s)",
        ),
        (
            "gpu-cost",
            "Sequences per client update",
            "Estimated trainer GPU USD / million input tokens",
        ),
    ]
    for name, xlabel, ylabel in plots:
        figure, axes = pyplot.subplots(
            1, len(contexts), figsize=(14, 4.5), squeeze=False, layout="constrained"
        )
        for axis, context in zip(axes[0], contexts):
            for device, tp, cp in layouts:
                rows = sorted(
                    [
                        (r["batch_size"], p)
                        for r in data["runs"]
                        if (r["hardware"], r["tp"], r["cp"]) == (device, tp, cp)
                        for p in r["points"]
                        if p["context_length"] == context
                    ],
                    key=lambda row: row[0],
                )
                if not rows:
                    continue
                batch = [r[0] for r in rows]
                latency = [r[1]["median_step_s"] for r in rows]
                if name == "latency-throughput":
                    throughput = [
                        r[1]["metrics"]["aggregate.training_input_tokens_per_s"] / 1000
                        for r in rows
                    ]
                    axis.plot(
                        latency,
                        throughput,
                        marker="o",
                        label=f"{device} · TP{tp}/CP{cp}",
                    )
                    for b, x, y in zip(batch, latency, throughput):
                        axis.annotate(
                            f" b={b}",
                            (x, y),
                            xytext=(4, 4),
                            textcoords="offset points",
                            fontsize=9,
                        )
                elif name == "step-latency":
                    axis.errorbar(
                        batch,
                        latency,
                        yerr=[
                            [v - min(r[1]["step_s"]) for v, r in zip(latency, rows)],
                            [max(r[1]["step_s"]) - v for v, r in zip(latency, rows)],
                        ],
                        marker="o",
                        capsize=3,
                        label=f"{device} · TP{tp}/CP{cp}",
                    )
                else:
                    cost = [
                        r[1]["metrics"][
                            "cost.estimated_gpu_usd_per_million_input_tokens"
                        ]
                        for r in rows
                    ]
                    axis.plot(
                        batch, cost, marker="o", label=f"{device} · TP{tp}/CP{cp}"
                    )
                if name != "latency-throughput":
                    axis.set_xticks(sorted({r["batch_size"] for r in data["runs"]}))
            axis.set(
                title=f"{context // 1024}K tokens / sequence",
                xlabel=xlabel,
                ylabel=ylabel,
            )
            axis.grid(alpha=0.25)
            axis.set_ylim(bottom=0)
            axis.margins(x=0.15, y=0.15)
            axis.legend()
        note = "5 measured updates after 2 warmups."
        if name == "step-latency":
            note += " Bars show observed min/max."
        if name == "gpu-cost":
            note = "GPU-only estimates at published rates; excludes startup, idle, CPU and memory."
        figure.suptitle(
            f"Qwen3.5-4B FFT · 8 allocated GPUs · one client\n{note}", fontsize=12
        )
        for suffix in ("svg", "png"):
            figure.savefig(args.measurements.with_name(f"{name}.{suffix}"), dpi=160)
        pyplot.close(figure)


if __name__ == "__main__":
    main()
