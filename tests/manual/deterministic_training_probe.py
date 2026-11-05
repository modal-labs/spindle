"""Fixed-input full-model replay; intentionally excludes rollout randomness."""

import argparse
import json
from pathlib import Path

import torch
from tinker import AdamParams, Datum, LoraConfig, ModelInput, TensorData

from spindle.backends.contract import ForwardBatch, ForwardItem, ModelSpec
from spindle.backends.miles_config import MilesBackendConfig
from spindle.backends.miles_lora import MilesCommandBackend
from spindle.backends.miles_runtime import runtime
from spindle.engine.spmd import _serialize_forward_output


def datum(step, length, offset=0):
    tokens = [100 + (i * 13 + step * 7 + offset) % 1000 for i in range(length + 1)]
    return Datum(
        model_input=ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=tokens[1:], dtype="int64", shape=[length]),
            "weights": TensorData(
                data=[0.0] * (length // 2) + [1.0 / length] * (length - length // 2),
                dtype="float32",
                shape=[length],
            ),
        },
    )


def main(args):
    from transformers import AutoConfig

    hf = AutoConfig.from_pretrained(args.checkpoint)
    hf = getattr(hf, "text_config", hf)
    gdn = "qwen3_5" in hf.model_type
    options = {
        "num_layers": hf.num_hidden_layers,
        "hidden_size": hf.hidden_size,
        "ffn_hidden_size": hf.intermediate_size,
        "num_attention_heads": hf.num_attention_heads,
        "num_query_groups": hf.num_key_value_heads,
        "kv_channels": hf.head_dim,
        "group_query_attention": True,
        "normalization": "RMSNorm",
        "layernorm_epsilon": hf.rms_norm_eps,
        "position_embedding_type": "rope",
        "rotary_base": int(
            getattr(hf, "rope_theta", None) or hf.rope_parameters["rope_theta"]
        ),
        "swiglu": True,
        "add_bias_linear": False,
        "qk_layernorm": True,
        "untie_embeddings_and_output_weights": not hf.tie_word_embeddings,
        "seed": 1234,
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
    }
    config = MilesBackendConfig(
        hf_checkpoint=args.checkpoint,
        model_type="qwen3.5-4B" if gdn else "",
        actor_num_gpus_per_node=args.tp,
        tensor_model_parallel_size=args.tp,
        max_lora_slots=2,
        max_lora_rank=8,
        default_lora_alpha=8,
        max_tokens_per_gpu=768,
        target_modules=("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"),
        deterministic_training=args.deterministic,
        deterministic_gdn=args.deterministic and gdn,
        deterministic_pack_clients=args.pack_clients,
        tp_reduce_precision="float64" if args.deterministic else None,
        extra_args=("--seq-length", "2048"),
        cli_options=options,
    )
    original = runtime._configure_actor_spec

    def configure(specs):
        original(specs)
        compute = specs._compute_spec_trainer

        def with_probe(*pos, **kw):
            spec = compute(*pos, **kw)
            if (
                spec.worker_class
                == "spindle.backends.miles_runtime.actor.SpindleMilesTrainRayActor"
            ):
                return spec.model_copy(
                    update={
                        "worker_class": "deterministic_probe_actor.DeterminismProbeActor"
                    }
                )
            return spec

        specs._compute_spec_trainer = with_probe

    runtime._configure_actor_spec = configure
    root = Path(args.output).parent / args.mode
    root.mkdir(parents=True, exist_ok=True)
    backend = MilesCommandBackend(
        config,
        checkpoint_dir=root / "checkpoints",
        capture_dir=root / "captures",
        base_model=args.model,
    )
    report = {
        "model": args.model,
        "mode": args.mode,
        "deterministic": args.deterministic,
        "pack_clients": args.pack_clients,
        "tp": args.tp,
        "steps": [],
        "torch": torch.__version__,
        "miles": backend.runtime.revision,
        "checkpoint_revision": Path(args.checkpoint).name,
    }

    def snapshot():
        result = backend.runtime._run(
            backend.runtime._trainer._execute_slots(
                "determinism_snapshot", slot=backend.job_to_slot["target"]
            )
        )
        return sorted(result, key=lambda row: row["rank"])

    def save():
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")

    try:
        for name in ("target", "neighbor"):
            backend.accept_model(
                name,
                ModelSpec(
                    base_model=args.model,
                    parameterization="lora",
                    lora_config=LoraConfig(rank=8, train_unembed=False),
                ),
            )
        report["initial"] = snapshot()
        # Each run starts from exactly the same checkpoint, even in the control.
        initial = Path(args.checkpoint_state)
        if initial.exists():
            backend.load_checkpoint("target", str(initial), restore_optimizer=True)
        else:
            backend.capture_checkpoint(
                "target", "initial", destination="initial", include_optimizer=True
            )
            initial = Path(backend.persist_checkpoint("initial", "initial"))
            report["checkpoint_state"] = str(initial)
            # A load materializes lazy Adam moments. Give the reference the
            # identical checkpoint round-trip before comparing optimizer state.
            backend.load_checkpoint("target", str(initial), restore_optimizer=True)
        report["restored"] = snapshot()
        if args.mode == "restore":
            midpoint = json.loads(Path(args.reference).read_text())[
                "midpoint_checkpoint"
            ]
            backend.load_checkpoint("target", midpoint, restore_optimizer=True)
            report["restored_midpoint"] = snapshot()
        save()
        start = args.steps // 2 if args.mode == "restore" else 0

        def trajectory(start, checkpoint_midpoint):
            for step in range(start, args.steps):
                lengths = (127, 257, 513) if step % 2 else (513, 127, 257)
                items = [
                    ForwardItem(
                        "target", tuple(datum(step, length) for length in lengths)
                    )
                ]
                if args.mode == "packed":
                    # Alternating neighbor sizes change the shared microbatch count.
                    items.insert(
                        0,
                        ForwardItem(
                            "neighbor",
                            tuple(
                                datum(step, 129 + i * 32, 71)
                                for i in range(1 + step % 5)
                            ),
                        ),
                    )
                batch = ForwardBatch(
                    items=tuple(items), loss_fn="cross_entropy", loss_fn_config={}
                )
                outputs = backend.forward_backward(batch)
                schedule = backend.runtime._run(
                    backend.runtime._trainer._execute_slots("determinism_schedule")
                )
                target = outputs[-1]
                before = snapshot()
                if any(row["gradients"]["tensors"] == 0 for row in before):
                    raise AssertionError("No gradients captured")
                names = tuple(item.model_id for item in items)
                optimizers = backend.optim_step(names, AdamParams(learning_rate=1e-4))
                if optimizers[-1].metrics["update_successful:mean"] != 1:
                    raise AssertionError("Optimizer update was skipped")
                after = snapshot()
                if all(
                    left["weights"] == right["weights"]
                    for left, right in zip(before, after, strict=True)
                ):
                    raise AssertionError("Optimizer did not change adapter weights")
                row = {
                    "step": step + 1,
                    "schedule": schedule,
                    "forward": _serialize_forward_output(target),
                    "before_optimizer": before,
                    "optimizer": {
                        key: value
                        for key, value in optimizers[-1].metrics.items()
                        if not key.startswith("timing/")
                    },
                    "after_optimizer": after,
                }
                report["steps"].append(row)
                if checkpoint_midpoint and step + 1 == args.steps // 2:
                    backend.capture_checkpoint(
                        "target",
                        "midpoint",
                        destination="midpoint",
                        include_optimizer=True,
                    )
                    report["midpoint_checkpoint"] = backend.persist_checkpoint(
                        "midpoint", "midpoint"
                    )
                    report["midpoint"] = snapshot()
                save()
                print("DETERMINISM_STEP", args.mode, step + 1, flush=True)
            (evaluation,) = backend.forward_backward(
                ForwardBatch(
                    items=(ForwardItem("target", (datum(99, 257),)),),
                    loss_fn="cross_entropy",
                    loss_fn_config={},
                    forward_only=True,
                )
            )
            report["final_evaluation"] = _serialize_forward_output(evaluation)
            scores = evaluation.loss_fn_outputs[0]["logprobs"].data
            report["final_nll"] = -sum(scores) / len(scores)
            if args.mode == "packed" and args.deterministic:
                mixed = any(
                    rank["mixed"] for row in report["steps"] for rank in row["schedule"]
                )
                if mixed != args.pack_clients:
                    raise AssertionError(
                        "Observed client mixing does not match the selected policy"
                    )
            save()

        trajectory(start, True)
        if args.mode == "reference":
            baseline = report
            backend.load_checkpoint("target", str(initial), restore_optimizer=True)
            report = {
                **baseline,
                "mode": "same_process",
                "steps": [],
                "restored": snapshot(),
            }
            trajectory(0, False)
            continued = report
            report = baseline
            report["same_process"] = continued
            save()
    finally:
        save()
        backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-state", required=True)
    parser.add_argument("--reference", default="")
    parser.add_argument(
        "--mode", choices=["reference", "repeat", "packed", "restore"], required=True
    )
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--pack-clients", action="store_true")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
