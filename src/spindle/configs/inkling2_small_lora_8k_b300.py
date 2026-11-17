"""Experimental Inkling-2-Small attention and unembedding LoRA; experts stay frozen.

Create stock Tinker clients with train_mlp=False. Training and sampling each
use a separate eight-B300 node.
"""

from spindle.configuration import BaseConfig


class Config(BaseConfig):
    name = "inkling2-small-lora-8k-b300"
    model = "thinkingmachines/Inkling-2-Small"
    weights_revision = "1df2d97d45210d94f781f42a6521392e5a539e46"
    trainer_image = "miles-v0.2.0-e186d97-inkling2-attn"
    inference_image = "sglang-inkling2-a2e3c49-head"
    max_context_length = 8192
    trainer_gpu = "B300"
    trainer_gpus_per_node = 8
    trainer_cpu = 64
    trainer_memory_mib = 1_200_000
    trainer_disk_mib = 1_500_000
    trainer_timeout_s = 172800
    trainer_max_clients_per_instance = 2
    miles_cfg = {
        "model_type": "inkling2",
        "megatron_to_hf_mode": "raw",
        "tensor_model_parallel_size": 1,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
        "max_lora_slots": 2,
        "max_lora_rank": 32,
        "default_lora_alpha": 32,
        "max_tokens_per_gpu": 8192,
        "target_modules": [
            "model.llm.layers.*.attn.wq_du",
            "model.llm.layers.*.attn.wr_du",
            "model.llm.layers.*.attn.gate_attn",
            "model.llm.layers.*.attn.wk_dv",
            "model.llm.layers.*.attn.wv_dv",
            "model.llm.layers.*.attn.wo_ud",
            "model.llm.unembed",
        ],
        "cli_options": {
            "lora_type": "canonical_lora",
            "qkv_format": "thd",
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
            "moe_shared_expert_overlap": False,
            "overlap_grad_reduce": False,
            "overlap_param_gather": False,
            "no_load_optim": True,
            "no_load_rng": True,
            "finetune": True,
        },
    }
    trainer_env = {
        "CUDA_MODULE_LOADING": "EAGER",
        "TORCH_NCCL_BLOCKING_WAIT": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
    inference_gpu = "B300"
    inference_gpus_per_node = 8
    inference_cpu = 32
    inference_memory_mib = 1_200_000
    inference_disk_mib = 1_000_000
    inference_startup_timeout_s = 7200
    inference_max_replicas = 1
    sglang_cfg = {
        "quantization": "inkling2",
        # The embedded legacy tag hides hf_quant_config.json's per-parameter precision.
        "json_model_override_args": '{"quantization_config": null}',
        "tp_size": 8,
        "ep_size": 8,
        "attention_backend": "fa4",
        "bf16_gemm_backend": "torch",
        "page_size": 128,
        "mem_fraction_static": 0.6,
        "max_running_requests": 16,
        "max_queued_requests": 16,
        "max_loaded_loras": 4,
        "max_loras_per_batch": 1,
        "lora_backend": "triton",
        "lora_strict_loading": True,
    }


config = Config()
