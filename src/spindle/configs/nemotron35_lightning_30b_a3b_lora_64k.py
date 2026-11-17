from spindle.configs.nemotron3_nano_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    name = "nemotron35-lightning-30b-a3b-lora-64k"
    trainer_image = "miles-v0.1.1-fdd808c"
    inference_image = "sglang-v0.5.23-c4182b2"
    model = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    trainer_gpu = "B300"
    trainer_gpus_per_node = 2
    trainer_cpu = 32
    trainer_memory_mib = 262144
    trainer_max_clients_per_instance = 4
    inference_max_replicas = 2
    inference_target_concurrency = 89
    overrides = {
        "miles_cfg.tensor_model_parallel_size": 1,
        "miles_cfg.expert_model_parallel_size": 1,
        "miles_cfg.max_lora_slots": 4,
        "miles_cfg.max_tokens_per_gpu": 65536,
        "miles_cfg.cli_options.log_probs_chunk_size": 4096,
        "miles_cfg.cli_options.recompute_method": "block",
        "miles_cfg.cli_options.recompute_num_layers": 36,
        "trainer_env.EXPERT_GROUPED_GEMM": "1",
        "sglang_cfg.max_loras_per_batch": 4,
        "sglang_cfg.max_loaded_loras": 32,
        "sglang_cfg.max_mamba_cache_size": 448,
        "sglang_cfg.mem_fraction_static": 0.9,
        "sglang_cfg.prefill_decode_interval": 16,
        "sglang_cfg.prefill_decode_interval_min_bs": 32,
        "sglang_cfg.lora_use_virtual_experts": True,
        "sglang_cfg.speculative_algorithm": "NEXTN",
        "sglang_cfg.speculative_num_steps": 3,
        "sglang_cfg.speculative_eagle_topk": 1,
        "sglang_cfg.speculative_num_draft_tokens": 4,
    }


config = Config()
