"""Modal app builders shared by all Python-configured model deployments.

Trainers and inference provisioners are independently deployed apps. Pools are
created on demand with the existing LoRA/FFT pool lifecycle.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from typing import Any, TypedDict, cast

import modal
import modal.experimental
from modal.config import config

from spindle.deployments import DeploymentConfig
from spindle.inference.serving import (
    SGLANG_ENV,
    sglang_busy,
    start_fft_sidecar,
    start_lora_sidecar,
    supervise_children,
    terminate,
    wait_http,
)
from spindle.startup_timing import SAMPLER_COMPONENT, TRAINER_COMPONENT, timeline
from spindle.usage import ContainerUsage, poll_busy

from .app_tags import app_tags
from . import (
    deepseek_v41_image,
    glm53_image,
    miles_v011_image,
    miles_v012_image,
    miles_v020_image,
    rollout_image,
)
from .deployment import trainer_deployment_env
from .fft_pool import FFTPoolSpec
from .fft_pool import deploy_pool as deploy_fft
from .image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)
from .kernel_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_ROOT,
    kernel_cache_volume,
)
from .kv import shared_kv
from .lora_pool import LoraPoolSpec
from .lora_pool import deploy_pool as deploy_lora
from .megatron_image import image as megatron_image
from .ray_cluster import start_trainer_cluster
from .serve import run_engine_with_backend
from .usage import USAGE_ROOT, sample_meter, start_usage_meter, usage_volume

# Named images are frozen: add a new name instead of editing one. See docs/deployment-configs.md#images.
# Each entry notes the fork PR that last changed its commit.
TRAINER_IMAGES = {
    # https://github.com/modal-projects/miles/pull/16 Support Tinker SDK 0.29+ in Miles' Tinker layer
    "miles-v0.1.1-fdd808c": miles_v011_image.image_01_fdd808c,
    # https://github.com/modal-projects/miles/pull/42 Support per-adapter seeds in the v0.1.1 LoRA slot API
    "miles-v0.1.1-52a7a99": miles_v011_image.image_02_52a7a99,
    # https://github.com/modal-projects/miles/pull/30 Prefetch the DeepSeek-V4.1 Engram tables at load
    "miles-v0.1.2-f2f8447": miles_v012_image.image_01_f2f8447,
    # https://github.com/modal-projects/miles/pull/43 Support per-adapter seeds in the v0.1.2 LoRA slot API (on top of #31)
    "miles-v0.1.2-99e1ebe": miles_v012_image.image_02_99e1ebe,
    # https://github.com/modal-projects/miles/pull/31 Support GLM-5.3 with a pinned provider dependency
    "miles-v0.1.2-464efc9-glm53": miles_v012_image.glm53_image_01_464efc9,
    # https://github.com/modal-projects/miles/pull/43 Support per-adapter seeds in the v0.1.2 LoRA slot API
    "miles-v0.1.2-99e1ebe-glm53": miles_v012_image.glm53_image_02_99e1ebe,
    # https://github.com/modal-labs/spindle/pull/307 miles#43 image with the GLM indexer scored in query blocks
    "miles-v0.1.2-99e1ebe-kpool-glm53": miles_v012_image.glm53_kpool_image_01_99e1ebe,
    # https://github.com/modal-projects/miles/pull/44 Support per-adapter seeds in the v0.2.0 LoRA slot API (on top of #37, Beam LoRA)
    "miles-v0.2.0-8c5de26": miles_v020_image.image_02_8c5de26,
    # https://github.com/modal-projects/miles/pull/49 Native canonical and routed-expert adapters, including Nemotron Nano
    "miles-v0.2.0-e814245": miles_v020_image.image_03_e814245,
    # Upstream NVIDIA/Megatron-LM commit; no Spindle fork
    "megatron-2d1fa8d": megatron_image,
}
SAMPLER_IMAGES = {
    # https://github.com/modal-projects/sglang/pull/104 Exact speculative top-k replay in mixed scoring batches
    "sglang-v0.5.23-3db151a": rollout_image.image_04_3db151a,
    "sglang-v0.5.23-e003cf9": rollout_image.image_03_e003cf9,
    # https://github.com/modal-projects/sglang/pull/81 Serve DeepSeek-V4 LoRA on quantized bases
    "sglang-v0.5.21-a3e0e17": deepseek_v41_image.image_01_a3e0e17,
    # https://github.com/modal-projects/sglang/pull/98 Honor the earliest stop in speculative token blocks
    "sglang-v0.5.23-c4182b2": rollout_image.image_01_c4182b2,
    # https://github.com/modal-projects/sglang/pull/98 Honor the earliest stop in speculative token blocks, with GLM-5.3 Flash sampler settings
    "sglang-v0.5.23-c4182b2-glm53": glm53_image.image_01_c4182b2,
}
# Used only by configs that leave trainer_image or inference_image empty.
DEFAULT_IMAGES = {
    "miles": "miles-v0.1.1-52a7a99",
    "megatron": "megatron-2d1fa8d",
    "sglang": "sglang-v0.5.23-c4182b2",
}


def image_for(backend: str, name: str = "") -> modal.Image:
    if not modal.is_local():
        return modal.Image.debian_slim()
    if ":" in name:
        module, attribute = name.rsplit(":", 1)
        return getattr(importlib.import_module(module), attribute)
    kind, images = (
        ("sampler", SAMPLER_IMAGES)
        if backend == "sglang"
        else ("trainer", TRAINER_IMAGES)
    )
    name = name or DEFAULT_IMAGES[backend]
    if name not in images:
        raise ValueError(f"unknown {kind} image {name!r}; choose from {sorted(images)}")
    return images[name]


def build_sampler_image(
    frontend: str, environment: str | None, images: Iterable[str] = ("",)
) -> None:
    """Build the pool image ahead of the first request; pools reuse the cached build."""
    app = modal.App.lookup(
        frontend, environment_name=environment, create_if_missing=True
    )
    for name in dict.fromkeys(images):
        image_for("sglang", name).build(app)


def volumes_for(platform: dict[str, Any]) -> dict[str, modal.Volume]:
    storage = platform["storage"]
    return {
        KERNEL_CACHE_ROOT: kernel_cache_volume,
        "/assets": modal.Volume.from_name(storage["assets"], create_if_missing=True),
        "/checkpoints": modal.Volume.from_name(
            storage["checkpoints"], create_if_missing=True, version=2
        ),
        "/bulletin": modal.Volume.from_name(
            storage["bulletin"], create_if_missing=True, version=2
        ),
        USAGE_ROOT: usage_volume(platform["frontend"]),
    }


def secrets_for(
    platform: dict[str, Any], *, training: bool = False
) -> list[modal.Secret]:
    names = platform["secrets"]
    result = [modal.Secret.from_name(names["api"], required_keys=["TINKER_API_KEY"])]
    if training:
        result.append(
            modal.Secret.from_name(
                names["sampler_proxy"],
                required_keys=["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"],
            )
        )
        if names["huggingface"]:
            result.append(modal.Secret.from_name(names["huggingface"]))
    return result


def trainer_experimental_options(
    rdma: bool, platform: dict[str, Any]
) -> dict[str, bool]:
    if not rdma:
        return {}
    options = {"efa_enabled": True}
    # Modal accepts this option only on clustered functions.
    if platform["modal"].get("exclude_reserved_capacity"):
        options["exclude_reserved_capacity"] = True
    return options


class ServerPlacement(TypedDict, total=False):
    routing_region: str
    compute_region: str


def server_placement(region: str | None) -> ServerPlacement:
    if region is None:
        return {}
    return {"routing_region": region, "compute_region": region}


def gpu_request(gpu: str | list[str], count: int) -> str | list[str]:
    """Modal tries list entries in order and schedules the first type with capacity."""
    if isinstance(gpu, str):
        return f"{gpu}:{count}"
    return [f"{name}:{count}" for name in gpu]


def log_scheduled_gpus(role: str) -> None:
    try:
        names = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        logging.getLogger(__name__).exception("query scheduled GPUs")
        return
    print(f"spindle {role} scheduled on {len(names)}x {', '.join(sorted(set(names)))}")


def deployment_env(values: Mapping[str, str]) -> dict[str, str]:
    """Keep user environment overrides separate from Spindle's deployment wiring."""
    if any(key.startswith("SPINDLE_") for key in values):
        raise ValueError("SPINDLE_ environment variables are managed by Spindle")
    return dict(values)


def build_trainer_app(
    deployment: DeploymentConfig,
    platform: dict[str, Any] | None = None,
    *,
    image: modal.Image | None = None,
) -> tuple[modal.App, modal.Function[[str, str, str], None, None]]:
    if platform is None:
        platform = cast(dict[str, Any], deployment.recipe.platform)
    recipe = deployment.recipe
    trainer_name = recipe.name
    app = modal.App(
        deployment.trainer_app_name,
        tags=app_tags("trainer", model=recipe.model, framework=recipe.backend),
    )

    env = {
        **trainer_deployment_env(),
        **deployment_env(recipe.trainer_env),
        "SPINDLE_APP_NAME": platform["frontend"],
    }

    def trainer(instance_id: str, config_json: str, frontend_app_id: str) -> None:
        timeline(TRAINER_COMPONENT, instance_id=instance_id).mark(
            "trainer_function_entered"
        )
        saved = DeploymentConfig.model_validate_json(config_json)
        if saved.name != trainer_name:
            raise ValueError("trainer settings do not match the deployed app")
        os.environ["SPINDLE_FRONTEND_APP_ID"] = frontend_app_id
        log_scheduled_gpus("trainer")
        run_trainer(saved, instance_id, platform)

    rdma = recipe.trainer_nodes > 1 or recipe.trainer_rdma
    if rdma:
        trainer = modal.clustered(size=recipe.trainer_nodes, rdma=True)(trainer)
    trainer = app.function(
        name="trainer",
        serialized=True,
        image=(
            image
            if image is not None
            else image_for(recipe.backend, recipe.trainer_image)
        ),
        gpu=gpu_request(recipe.trainer_gpu, recipe.trainer_gpus_per_node),
        region=platform["modal"]["region"],
        cpu=recipe.trainer_cpu,
        memory=recipe.trainer_memory_mib,
        ephemeral_disk=recipe.trainer_disk_mib or None,
        timeout=recipe.trainer_timeout_s,
        max_containers=None,
        min_containers=0,
        single_use_containers=True,
        volumes={**volumes_for(platform)},
        secrets=secrets_for(platform, training=True),
        env={**env},
        experimental_options=trainer_experimental_options(rdma, platform),
    )(trainer)

    return app, trainer


def run_trainer(
    deployment: DeploymentConfig,
    instance_id: str,
    platform: dict[str, Any] | None = None,
) -> None:
    if platform is None:
        platform = cast(dict[str, Any], deployment.recipe.platform)
    recipe = deployment.recipe
    settings = deployment.trainer_settings
    assets = volumes_for(platform)["/assets"]
    startup = timeline(TRAINER_COMPONENT, instance_id=instance_id)
    if recipe.trainer_nodes > 1:
        ray_address = start_trainer_cluster(
            recipe.trainer_nodes,
            before_head=assets.reload,
            before_worker_join=assets.reload,
        )
        if ray_address is None:
            return
        startup.mark("ray_cluster_joined")
    else:
        assets.reload()
        ray_address = None
        startup.mark("assets_reloaded")
    if recipe.parameterization == "lora":
        _warm_lora_pool(platform["frontend"], deployment.definition_id)
        startup.mark("lora_pool_warm_requested")
    env = {
        **KERNEL_CACHE_ENV,
        **deployment_env(recipe.trainer_env),
        "SPINDLE_APP_NAME": platform["frontend"],
        "SPINDLE_BACKEND_CONFIG": json.dumps(settings),
        "SPINDLE_BASE_MODEL": recipe.model,
        "SPINDLE_DEFINITION_ID": deployment.definition_id,
        "SPINDLE_CHECKPOINT_VOLUME": platform["storage"]["checkpoints"],
        "SPINDLE_BULLETIN_ROOT": "/bulletin",
        "SPINDLE_BULLETIN_VOLUME": platform["storage"]["bulletin"],
        "SPINDLE_DEFINITION_REVISION": deployment.definition_id,
    }
    if ray_address is not None:
        env["SPINDLE_RAY_ADDRESS"] = ray_address
    executor = (
        "spindle.backends.miles_lora:build_executor"
        if recipe.backend == "miles"
        else "spindle.backends.megatron_fft:build_executor"
    )

    async def failed(error: Exception) -> None:
        await shared_kv().put(
            f"deployment_failure:{deployment.definition_id}",
            {"error": str(error), "instance_id": instance_id, "failed_at": time.time()},
        )

    run_engine_with_backend(
        shared_kv(),  # ty: ignore[invalid-argument-type]
        executor,
        definition_id=deployment.definition_id,
        revision=config["image_id"],
        instance_id=instance_id,
        backend_env=env,
        nproc=1 if recipe.backend == "miles" else recipe.trainer_gpus_per_node,
        max_models=recipe.trainer_max_clients_per_instance,
        sampler_persistence_concurrency=recipe.sampler_persistence_concurrency,
        max_forward_backward_models=recipe.trainer_max_clients_per_batch,
        on_startup_error=failed,
        usage=lambda: ContainerUsage(
            start_usage_meter(platform["frontend"]),
            app_name=deployment.trainer_app_name,
            base_model=recipe.model,
            started_at=startup.origin,
        ),
    )


def _warm_lora_pool(frontend: str, definition_id: str) -> None:
    """Deploy the sampler pool now that a trainer has GPUs, not while it queues."""
    try:
        modal.Function.from_name(frontend, "ensure_lora_pool").spawn(
            LoraPoolSpec(definition_id, frontend).as_dict()
        )
    except Exception:
        logging.getLogger(__name__).exception("warm LoRA pool %s", definition_id)


def build_rollout_app(
    deployment: DeploymentConfig,
    pool: LoraPoolSpec | FFTPoolSpec,
    platform: dict[str, Any] | None = None,
    *,
    image: modal.Image | None = None,
) -> tuple[modal.App, modal.Server]:
    """Create one frozen-base LoRA pool or one FFT latest/pinned/base pool."""
    if platform is None:
        platform = cast(dict[str, Any], deployment.recipe.platform)
    recipe = deployment.recipe
    lora = recipe.parameterization == "lora"
    if pool.definition_id != deployment.definition_id:
        raise ValueError("pool does not match deployment")
    app = modal.App(
        pool.app_name,
        tags=app_tags("inference", model=recipe.model, framework="sglang"),
    )
    options = deployment.inference_settings
    minimum = recipe.inference_min_replicas
    maximum = recipe.inference_max_replicas
    window = recipe.inference_scaledown_window_s
    if isinstance(pool, FFTPoolSpec):
        minimum = minimum if pool.min_containers is None else pool.min_containers
        maximum = maximum if pool.max_containers is None else pool.max_containers
        window = window if pool.scaledown_window is None else pool.scaledown_window

    @app.server(
        name="Server",
        serialized=True,
        image=image
        if image is not None
        else image_for("sglang", recipe.inference_image),
        gpu=gpu_request(recipe.inference_gpu, recipe.inference_gpus_per_node),
        cpu=recipe.inference_cpu,
        memory=recipe.inference_memory_mib,
        ephemeral_disk=recipe.inference_disk_mib or None,
        volumes={**volumes_for(platform)},
        secrets=secrets_for(platform),
        env=dict(
            deployment_env(
                SGLANG_ENV
                # Split-K speeds up decode-sized LoRA batches on H200 and B200.
                | ({"SGLANG_ENABLE_LORA_CSGMV_SPLIT_K": "1"} if lora else {})
                | recipe.inference_env
                | (
                    {
                        "SGLANG_SPINDLE_ROUTING_OBSERVATIONS": "1",
                    }
                    if lora and recipe.inference_lora_routing == "cache_wait"
                    else {}
                )
            ),
            SPINDLE_LORA_UPDATE_IN_PLACE="1"
            if recipe.inference_lora_update_in_place
            else "0",
        ),
        min_containers=minimum,
        max_containers=maximum,
        target_concurrency=recipe.inference_target_concurrency
        or options["max_running_requests"],
        scaleup_window=recipe.inference_scaleup_window_s,
        scaledown_window=window,
        startup_timeout=recipe.inference_startup_timeout_s,
        exit_grace_period=300,
        port=8000,
        **server_placement(platform["modal"]["region"]),
    )
    class Server:
        @modal.enter()
        def start(self) -> None:
            self.sidecar = None
            self.sglang = None
            self.busy_poll = None
            log_scheduled_gpus("sampler")
            startup = timeline(SAMPLER_COMPONENT, pool=pool.app_name)
            startup.mark("server_enter")

            self.sglang = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "spindle.inference.sglang",
                    deployment.asset_path,
                    json.dumps(options),
                ],
                start_new_session=True,
            )
            try:
                wait_http(
                    "http://127.0.0.1:8001/health",
                    self.sglang,
                    recipe.inference_startup_timeout_s,
                )
                startup.mark("sglang_healthy")
                kwargs = dict(
                    port=8000,
                    sglang_port=8001,
                    bulletin_root="/bulletin",
                    bulletin_volume=platform["storage"]["bulletin"],
                )
                self.sidecar = (
                    start_lora_sidecar(**kwargs)  # ty: ignore[invalid-argument-type]
                    if lora
                    else start_fft_sidecar(
                        **kwargs,  # ty: ignore[invalid-argument-type]
                        run_id=pool.model_id,
                        pinned_version=None if pool.latest else pool.version,
                    )
                )
                self.supervisor = supervise_children(self.sglang, self.sidecar)
                wait_http(
                    "http://127.0.0.1:8000/health",
                    self.sidecar,
                    recipe.inference_startup_timeout_s,
                )
                startup.mark("sidecar_healthy")
            except BaseException:
                terminate(self.sidecar)
                terminate(self.sglang)
                raise
            if (meter := sample_meter(platform["frontend"])) is not None:
                busy = sglang_busy(8001)
                try:
                    wait_http("http://127.0.0.1:8001/metrics", self.sglang, timeout=30)
                    busy()
                except Exception:
                    # Without busy time, the pool's cost is split by tokens alone.
                    logging.getLogger(__name__).exception(
                        "Can't read SGLang metrics; not recording pool idle time"
                    )
                else:
                    usage = ContainerUsage(
                        meter,
                        app_name=pool.app_name,
                        base_model=recipe.model,
                        started_at=startup.origin,
                    )
                    self.busy_poll = poll_busy(usage, busy)

        @modal.exit()
        def stop(self) -> None:
            if self.busy_poll is not None:
                self.busy_poll.set()
            terminate(self.sidecar)
            terminate(self.sglang)

    return app, Server


def build_inference_app(
    deployment: DeploymentConfig,
    platform: dict[str, Any] | None = None,
    *,
    image: modal.Image | None = None,
) -> tuple[modal.App, modal.Function[[str, dict[str, Any]], str, str]]:
    """Deploy the code that provisions this configuration's inference pools."""
    if platform is None:
        platform = cast(dict[str, Any], deployment.recipe.platform)

    if image is None:
        image = (
            modal.Image.debian_slim(python_version="3.12")
            .apt_install("git")
            .pip_install(
                *CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE, "huggingface-hub"
            )
            .add_local_python_source("spindle", copy=True, ignore=ignore_config_source)
        )
    app = modal.App(
        deployment.inference_app_name,
        tags=app_tags("inference-provisioner", model=deployment.recipe.model),
    )
    inference_name = deployment.name

    @app.function(name="provision", image=image, serialized=True, timeout=1800)
    def provision(config_json: str, pool_data: dict[str, Any]) -> str:
        saved = DeploymentConfig.model_validate_json(config_json)
        if saved.name != inference_name:
            raise ValueError("inference settings do not match the deployed app")
        if pool_data["definition_id"] != saved.definition_id:
            raise ValueError("pool definition does not match deployment")
        if saved.parameterization == "lora":
            return deploy_lora(
                LoraPoolSpec.from_dict(pool_data), config=saved, platform=platform
            )
        return deploy_fft(
            FFTPoolSpec.from_dict(pool_data), config=saved, platform=platform
        )

    return app, provision
