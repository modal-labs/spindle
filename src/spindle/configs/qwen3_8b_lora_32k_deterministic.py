"""Experimental fixed-topology deterministic trainer; validate before deployment."""

from spindle.configs.qwen3_8b_lora_32k import Config as Parent


class Config(Parent):
    name = "qwen3-8b-lora-32k-deterministic"
    miles_cfg = {
        **Parent.miles_cfg,
        "tensor_model_parallel_size": 4,
        "deterministic_training": True,
        "tp_reduce_precision": "float64",
    }
    trainer_env = {
        **Parent.trainer_env,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_ALGO": "Ring",
        "NCCL_PROTO": "Simple",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    }
    sglang_cfg = {**Parent.sglang_cfg, "enable_deterministic_inference": True}


config = Config()
