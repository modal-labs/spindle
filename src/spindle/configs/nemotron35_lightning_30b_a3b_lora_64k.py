from spindle.configs.nemotron3_nano_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    name = "nemotron35-lightning-30b-a3b-lora-64k"
    trainer_image = "miles-v0.1.1-fdd808c"
    inference_image = "sglang-v0.5.23-c4182b2"
    model = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    trainer_gpus_per_node = 2
    trainer_cpu = 32
    trainer_memory_mib = 262144
    trainer_max_clients_per_instance = 4
    # A replica takes about four minutes to start, so add a second one only when the first is full.
    inference_max_replicas = 2
    inference_target_concurrency = 128
    overrides = {
        "miles_cfg.expert_model_parallel_size": 2,
        "miles_cfg.max_lora_slots": 4,
        "miles_cfg.cli_options.log_probs_chunk_size": 4096,
        "sglang_cfg.max_loras_per_batch": 4,
        "sglang_cfg.max_loaded_loras": 32,
        "sglang_cfg.max_mamba_cache_size": 640,
        "sglang_cfg.mem_fraction_static": 0.9,
        # Prefill runs eagerly on this model, so batch admissions once 32 requests decode.
        "sglang_cfg.prefill_decode_interval": 16,
        "sglang_cfg.prefill_decode_interval_min_bs": 32,
    }


config = Config()
