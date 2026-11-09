import os

FRONTEND = os.environ.get("SPINDLE_VALIDATION_FRONTEND", "score-centering-validation")

OVERRIDES = {
    "max_context_length": 2048,
    "trainer_gpu": "B300",
    "trainer_nodes": 1,
    "trainer_cpu": 32,
    "trainer_memory_mib": 262144,
    "trainer_timeout_s": 7200,
    "inference_gpu": "B300",
    "inference_gpus_per_node": 1,
    "inference_cpu": 16,
    "inference_memory_mib": 65536,
    "inference_min_replicas": 0,
    "inference_max_replicas": 1,
    "inference_target_concurrency": 8,
    "inference_scaledown_window_s": 600,
    "platform.requires_proxy_auth": True,
    "sglang_cfg.tp_size": 1,
    "sglang_cfg.max_running_requests": 8,
    "sglang_cfg.max_queued_requests": 32,
    "sglang_cfg.mem_fraction_static": 0.7,
    "trainer_gpus_per_node": 8,
}
OVERRIDES.update(
    {
        "platform.frontend": FRONTEND,
        "platform.modal.environment": os.environ.get("MODAL_ENVIRONMENT", "main"),
        "platform.storage.checkpoints": f"{FRONTEND}-checkpoints",
        "platform.storage.bulletin": f"{FRONTEND}-bulletin",
    }
)
