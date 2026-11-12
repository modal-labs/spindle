"""Beam BF16 LoRA on sixteen H200 trainers and one eight-B300 sampler."""

from spindle.configuration import BaseConfig


class Config(BaseConfig):
    name = "beam-501b-a23b-lora-5k"
    model = "reflection/Beam-1.0-501B-A23B"
    weights_revision = "12950ae640b48cbbe9b5bb7735bf197507d6383f"
    max_context_length = 5120
    trainer_image = "miles-v0.2.0"
    trainer_gpu = "H200"
    trainer_gpus_per_node = 8
    trainer_nodes = 2
    trainer_cpu = 32
    trainer_memory_mib = 524288
    trainer_max_clients_per_instance = 2
    trainer_max_clients_per_batch = 1
    trainer_timeout_s = 86400
    miles_cfg = {
        "model_type": "beam",
        "tensor_model_parallel_size": 8,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 16,
        "expert_tensor_parallel_size": 1,
        "max_tokens_per_gpu": 5120,
        "dynamic_batch_size": False,
        "max_lora_slots": 2,
        "max_lora_rank": 16,
        "default_lora_alpha": 32,
        "target_modules": [
            "*.q_proj",
            "*.k_proj",
            "*.v_proj",
            "*.o_proj",
            "*.self_attn.gate_proj",
            "*.mlp.gate_proj",
            "*.mlp.up_proj",
            "*.mlp.down_proj",
            "*.mlp.experts.*.gate_proj",
            "*.mlp.experts.*.up_proj",
            "*.mlp.experts.*.down_proj",
        ],
        "cli_options": {
            "lora_type": "lora",
            "bf16": True,
            "qkv_format": "bshd",
            "attention_backend": "unfused",
            "data_pad_size_multiplier": 1,
            "moe_router_dtype": "fp32",
            "moe_router_use_torch_mm": True,
            "moe_router_bias_update_rate": 0,
            "moe_aux_loss_coeff": [0.0],
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
            "distributed_timeout_minutes": 120,
            "use_rollout_routing_replay": True,
        },
    }
    inference_gpu = "B300"
    inference_gpus_per_node = 8
    inference_cpu = 32
    inference_memory_mib = 524288
    inference_min_replicas = 1
    inference_max_replicas = 1
    inference_target_concurrency = 64
    inference_startup_timeout_s = 7200
    sglang_cfg = {
        "tp_size": 8,
        "ep_size": 1,
        "dtype": "bfloat16",
        "quantization": None,
        "attention_backend": "trtllm_mha",
        "moe_runner_backend": "triton",
        "lora_backend": "csgmv",
        "enable_return_routed_experts": True,
        "disable_radix_cache": True,
        "mem_fraction_static": 0.85,
        "max_running_requests": 64,
        "max_queued_requests": 256,
        "max_loaded_loras": 4,
        "max_loras_per_batch": 2,
        "chunked_prefill_size": 8192,
        "cuda_graph_max_bs_decode": 64,
        "model_loader_extra_config": {
            "enable_multithread_load": True,
            "num_threads": 2,
        },
    }


config = Config()
