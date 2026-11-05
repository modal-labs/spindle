"""Experimental Hopper hdim256 path ported from modal-projects/lilo#28."""

from spindle.configs.qwen35_9b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-lora-16k-deterministic"
    trainer_image = "miles-deterministic-fa3"
    miles_cfg = {
        **Parent.miles_cfg,
        "deterministic_training": True,
        "deterministic_gdn": True,
        "tp_reduce_precision": "float64",
    }
    trainer_env = {
        **Parent.trainer_env,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_ALGO": "Ring",
        "NCCL_PROTO": "Simple",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    }
    sglang_cfg = {
        **Parent.sglang_cfg,
        "enable_deterministic_inference": True,
        "attention_backend": "fa3",
    }


config = Config()
