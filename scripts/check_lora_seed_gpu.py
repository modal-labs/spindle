"""Check seeded initialization and training using the stock SDK and a real GPU backend.

Run with: modal run --env ENV scripts/check_lora_seed_gpu.py
Uses one temporary H100 and cached Qwen3-8B weights from spindle-model-assets.
For DeepSeek-V2-Lite, first run with --prepare, then --mla.
Set SPINDLE_SEED_MILES_API=0.2 to check the v0.2.0 image.
"""

import json
import os
from pathlib import Path
from types import ModuleType
from typing import TypedDict, cast

import modal

from spindle.backends.miles_config import MilesBackendConfig
from spindle.providers.modal.miles_seed_images import IMAGES

image = IMAGES[
    "miles-v0.2.0-384d73b"
    if os.environ.get("SPINDLE_SEED_MILES_API") == "0.2"
    else "miles-v0.1.1-ce494f0"
]


class SeedRun(TypedDict):
    seed: int | None
    slot: int
    initial: list[dict[str, str]]
    final: list[dict[str, str]]
    trajectory: list[list[float]]


app = modal.App("spindle-pr324-seed-check")
actor_path = Path(__file__).parents[1] / "tests/gpu/lora_seed_actor.py"
support_path = Path(__file__).parents[1] / "tests/support.py"


class BackendOptions(TypedDict):
    actor_num_gpus_per_node: int
    max_lora_slots: int
    max_lora_rank: int
    max_tokens_per_gpu: int


def _backend_config(model: str) -> MilesBackendConfig:
    common: BackendOptions = dict(
        actor_num_gpus_per_node=1,
        max_lora_slots=4,
        max_lora_rank=8,
        max_tokens_per_gpu=512,
    )
    if model == "Qwen/Qwen3-8B":
        return MilesBackendConfig(
            hf_checkpoint=f"/assets/{model}",
            model_type="qwen3-8B",
            target_modules=(
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
            **common,
        )
    if model != "deepseek-ai/DeepSeek-V2-Lite":
        raise ValueError(f"unsupported seed test model: {model}")
    root = Path("/seed-assets/DeepSeek-V2-Lite")
    hf = json.loads((root / "config.json").read_text())
    options = {
        "add_bias_linear": False,
        "num_layers": hf["num_hidden_layers"],
        "hidden_size": hf["hidden_size"],
        "ffn_hidden_size": hf["intermediate_size"],
        "num_attention_heads": hf["num_attention_heads"],
        "kv_channels": hf["qk_nope_head_dim"],
        "normalization": "RMSNorm",
        "position_embedding_type": "rope",
        "norm_epsilon": hf["rms_norm_eps"],
        "swiglu": True,
        "untie_embeddings_and_output_weights": True,
        "vocab_size": hf["vocab_size"],
        "seq_length": 512,
        "max_position_embeddings": hf["max_position_embeddings"],
        "multi_latent_attention": True,
        "kv_lora_rank": hf["kv_lora_rank"],
        "qk_head_dim": hf["qk_nope_head_dim"],
        "qk_pos_emb_head_dim": hf["qk_rope_head_dim"],
        "v_head_dim": hf["v_head_dim"],
        "qk_layernorm": True,
        "rotary_scaling_factor": hf["rope_scaling"]["factor"],
        "rotary_base": hf["rope_theta"],
        "mscale": hf["rope_scaling"]["mscale"],
        "mscale_all_dim": hf["rope_scaling"]["mscale_all_dim"],
        "num_experts": hf["n_routed_experts"],
        "moe_layer_freq": json.dumps(
            [0] * hf["first_k_dense_replace"]
            + [1] * (hf["num_hidden_layers"] - hf["first_k_dense_replace"])
        ),
        "moe_ffn_hidden_size": hf["moe_intermediate_size"],
        "moe_router_topk": hf["num_experts_per_tok"],
        "moe_shared_expert_intermediate_size": hf["moe_intermediate_size"]
        * hf["n_shared_experts"],
        "moe_router_pre_softmax": True,
        "moe_router_score_function": hf["scoring_func"],
        "moe_router_load_balancing_type": "seq_aux_loss",
        "moe_token_dispatcher_type": "alltoall",
        "moe_aux_loss_coeff": 0.0,
        "moe_router_group_topk": 1,
        "moe_router_num_groups": 1,
        "moe_grouped_gemm": True,
        "moe_router_topk_scaling_factor": hf["routed_scaling_factor"],
        "moe_router_dtype": "fp32",
        "moe_permute_fusion": True,
    }
    return MilesBackendConfig(
        hf_checkpoint=str(root),
        model_type="",
        cli_options=options,
        target_modules=(
            "q_proj",
            "kv_a_proj_with_mqa",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        **common,
    )


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install("huggingface-hub"),
    volumes={
        "/seed-assets": modal.Volume.from_name(
            "spindle-pr324-seed-assets", create_if_missing=True, version=2
        )
    },
    memory=8192,
    timeout=1800,
)
def prepare_mla() -> dict[str, str]:
    # The model downloader is only needed inside this preparation image.
    from huggingface_hub import HfApi, snapshot_download

    root = Path("/seed-assets/DeepSeek-V2-Lite")
    marker = root / "seed-validation-source.json"
    if marker.exists():
        return json.loads(marker.read_text())
    model = "deepseek-ai/DeepSeek-V2-Lite"
    revision = HfApi().model_info(model).sha
    assert revision is not None
    snapshot_download(
        model,
        revision=revision,
        local_dir=str(root),
        allow_patterns=["*.json", "*.safetensors", "*.model"],
    )
    assert list(root.glob("*.safetensors")), "checkpoint has no safetensors weights"
    config = json.loads((root / "config.json").read_text())
    # Use the image's built-in Transformers implementation for the same weights.
    config.pop("auto_map", None)
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    result = {
        "model": model,
        "revision": revision,
        "config_loader": "built-in Transformers DeepseekV2Config",
    }
    marker.write_text(json.dumps(result, indent=2) + "\n")
    modal.Volume.from_name("spindle-pr324-seed-assets").commit()
    return result


@app.function(
    image=image.add_local_file(actor_path, "/root/lora_seed_actor.py")
    .add_local_file(support_path, "/root/tests/support.py")
    .add_local_file(support_path.with_name("__init__.py"), "/root/tests/__init__.py"),
    gpu="H100",
    cpu=16,
    memory=98304,
    timeout=1800,
    volumes={
        "/assets": modal.Volume.from_name("spindle-model-assets"),
        "/seed-assets": modal.Volume.from_name(
            "spindle-pr324-seed-assets", create_if_missing=True, version=2
        ),
    },
    env={
        "PYTHONPATH": "/root:/root/miles:/root/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    },
)
def check(model: str = "Qwen/Qwen3-8B") -> dict[str, object]:
    # These imports require the GPU image rather than the local CLI environment.
    import gc
    from types import SimpleNamespace

    import httpx
    import numpy as np
    import tinker
    from tests.support import serve
    from tinker import types

    from spindle.backends.miles_lora import MilesCommandBackend
    from spindle.backends.miles_runtime import runtime
    from spindle.control_plane import ControlPlane, create_control_plane_app
    from spindle.engine.api import Executor
    from spindle.engine.backend_http import HttpBackendClient, create_backend_app
    from spindle.engine.spmd import DistributedExecutor
    from spindle.providers.local import InMemoryKeyValueStore, LocalEnginePlatform

    gc.disable()
    original_configure = runtime._configure_actor_spec

    def configure(specs: ModuleType) -> None:
        original_configure(specs)
        original = specs._compute_spec_trainer

        def compute(*args: object, **kwargs: object) -> object:
            spec = original(*args, **kwargs)
            if (
                spec.worker_class
                == "spindle.backends.miles_runtime.actor.SpindleMilesTrainRayActor"
            ):
                return spec.model_copy(
                    update={"worker_class": "lora_seed_actor.SeedProbeActor"}
                )
            return spec

        specs._compute_spec_trainer = compute

    if os.environ.get("SPINDLE_MILES_API") == "0.2":
        from spindle.backends.miles_runtime import v020

        v020.ACTOR_CLASS = "lora_seed_actor.SeedProbeActor"
    else:
        runtime._configure_actor_spec = configure
    backend = MilesCommandBackend(
        _backend_config(model),
        checkpoint_dir=Path("/tmp/seed-checkpoints"),
        capture_dir=Path("/tmp/seed-captures"),
        base_model=model,
    )
    definition = "seed-check"
    executor = HttpBackendClient(
        "http://backend",
        transport=httpx.ASGITransport(
            app=create_backend_app(cast(Executor, DistributedExecutor(backend)))
        ),
    )
    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(definition, lambda: executor, max_models=4),
    )
    definitions = (
        SimpleNamespace(
            definition_id=definition,
            name=definition,
            model=model,
            parameterization="lora",
            max_context_length=512,
        ),
    )
    api = create_control_plane_app(
        plane, definitions, api_key="tml-seed-test", retrieve_window=5
    )
    tokens = [100 + i % 70 for i in range(129)]
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(
                data=tokens[1:], dtype="int64", shape=[128]
            ),
            "weights": types.TensorData(data=[1.0] * 128, dtype="float32", shape=[128]),
        },
    )

    def probe(slot: int) -> list[dict[str, str]]:
        return backend.runtime._run(
            backend.runtime._trainer._execute_slots("seed_probe", slot=slot)
        )

    results: list[SeedRun] = []
    clients = []
    try:
        with serve(api) as url:
            service = tinker.ServiceClient(base_url=url, api_key="tml-seed-test")
            for index, seed in enumerate((0, 0, 1, 0, None)):
                before = probe(0)
                client = service.create_lora_training_client(
                    base_model=model, rank=8, seed=seed, train_unembed=False
                )
                clients.append(client)
                slot = backend.job_to_slot[client.model_id]
                initial = probe(slot)
                if "DeepSeek" in model:
                    assert all(
                        int(state["replicated_modules"]) > 0 for state in initial
                    )
                if seed is not None:
                    assert [x["rng"] for x in before] == [x["rng"] for x in initial], (
                        "initialization changed trainer RNG"
                    )
                trajectory: list[list[float]] = []
                for _ in range(3):
                    client.forward_backward([datum], "cross_entropy").result(
                        timeout=180
                    )
                    client.optim_step(types.AdamParams(learning_rate=1e-3)).result(
                        timeout=180
                    )
                    output = client.forward([datum], "cross_entropy").result(
                        timeout=180
                    )
                    trajectory.append(
                        [
                            float(value)
                            for value in output.loss_fn_outputs[0]["logprobs"].data
                        ]
                    )
                result = SeedRun(
                    seed=seed,
                    slot=slot,
                    initial=initial,
                    final=probe(slot),
                    trajectory=trajectory,
                )
                results.append(result)
                print("SEED_RUN", json.dumps(result), flush=True)
                if index in (1, 2, 3, 4):
                    unload = clients[0] if index == 1 else client
                    response = httpx.post(
                        url + "/api/v1/unload_model",
                        json={"model_id": unload.model_id},
                        headers={"x-api-key": "tml-seed-test"},
                        timeout=30,
                    )
                    response.raise_for_status()
            hashes = [[x["weights"] for x in run["initial"]] for run in results]
            assert hashes[0] == hashes[1] == hashes[3]
            assert hashes[0] != hashes[2]
            same = [
                float(
                    np.max(
                        np.abs(
                            np.array(results[0]["trajectory"])
                            - np.array(results[i]["trajectory"])
                        )
                    )
                )
                for i in (1, 3)
            ]
            different = float(
                np.max(
                    np.abs(
                        np.array(results[0]["trajectory"])
                        - np.array(results[2]["trajectory"])
                    )
                )
            )
            assert all(np.isfinite(run["trajectory"]).all() for run in results)
            final_hashes = [[x["weights"] for x in run["final"]] for run in results]
            assert final_hashes[0] == final_hashes[1] == final_hashes[3]
            assert max(same) <= 1e-5, same
            assert different > 1e-5, different
            summary = dict(
                model=model,
                miles_commit=backend.runtime.revision,
                miles_api=os.environ.get("SPINDLE_MILES_API", "0.1"),
                hardware="H100:1",
                steps=3,
                same_seed_max_logprob_error=same,
                different_seed_max_logprob_error=different,
                runs=results,
            )
            print("SEED_RESULT", json.dumps(summary), flush=True)
            return summary
    finally:
        backend.close()


@app.local_entrypoint()
def main(
    output: str = "/tmp/spindle-seed-gpu-result.json",
    mla: bool = False,
    prepare: bool = False,
) -> None:
    if prepare:
        print(json.dumps(prepare_mla.remote(), indent=2))
        return
    result = check.remote("deepseek-ai/DeepSeek-V2-Lite" if mla else "Qwen/Qwen3-8B")
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "runs"}, indent=2
        )
    )
