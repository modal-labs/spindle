from spindle.configs.nemotron3_nano_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    name = "nemotron35-lightning-30b-a3b-lora-64k"
    model = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    trainer_gpus_per_node = 2
    trainer_cpu = 32
    trainer_memory_mib = 262144
    trainer_max_clients_per_instance = 4
    overrides = {
        # Every GPU holds all 128 experts, so no tokens move between GPUs.
        "miles_cfg.expert_model_parallel_size": 1,
        "miles_cfg.max_lora_slots": 4,
        # Score long rows 4,096 tokens at a time. Scoring a 64K row in one pass
        # takes about 15 GB more on each GPU.
        "miles_cfg.cli_options.log_probs_chunk_size": 4096,
        "sglang_cfg.max_loras_per_batch": 4,
        "sglang_cfg.max_loaded_loras": 32,
        # Rows packed into one micro-batch must not share Mamba state.
        "trainer_env.MAMBA_SEQUENCE_RESET": "1",
    }


config = Config()
