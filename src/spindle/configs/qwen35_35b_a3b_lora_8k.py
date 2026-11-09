from spindle.configs.qwen36_35b_a3b_lora_32k import Config as Parent


class Config(Parent):
    name = "qwen35-35b-a3b-lora-8k"
    model = "Qwen/Qwen3.5-35B-A3B"
    weights_revision = "59d61f3ce65a6d9863b86d2e96597125219dc754"
    max_context_length = 8192
    trainer_gpu = "B300"
    trainer_gpus_per_node = 4
    trainer_cpu = 16
    trainer_memory_mib = 262144
    trainer_max_clients_per_instance = 1
    inference_gpu = "B300"
    inference_gpus_per_node = 1
    inference_memory_mib = 262144
    inference_max_replicas = 1
    inference_target_concurrency = 128
    inference_env = {"SGLANG_DEEPEP_BF16_DISPATCH": "1"}
    overrides = {
        "miles_cfg.model_type": "qwen3.5-35B-A3B_lora",
        "miles_cfg.expert_model_parallel_size": 4,
        "miles_cfg.max_lora_slots": 1,
        "miles_cfg.max_lora_rank": 16,
        "miles_cfg.default_lora_alpha": 16,
        "miles_cfg.max_tokens_per_gpu": 8192,
        "miles_cfg.target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "sglang_cfg.tp_size": 1,
        "sglang_cfg.ep_size": 1,
        "sglang_cfg.moe_runner_backend": "deep_gemm",
        "sglang_cfg.mem_fraction_static": 0.7,
        "sglang_cfg.max_running_requests": 128,
        "sglang_cfg.chunked_prefill_size": 4096,
        "sglang_cfg.cuda_graph_bs_decode": [1, 2, 4, 8, 16, 32, 64, 128],
        "sglang_cfg.cuda_graph_bs_prefill": [256, 512, 1024, 2048, 4096],
        "sglang_cfg.max_queued_requests": 128,
        "sglang_cfg.max_loaded_loras": 4,
        "sglang_cfg.max_loras_per_batch": 1,
        "sglang_cfg.lora_strict_loading": True,
    }


config = Config()
