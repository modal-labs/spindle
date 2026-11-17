from spindle.configs.nemotron3_super_120b_a12b_lora_64k_b300 import Config as Parent


class Config(Parent):
    name = "nemotron3-super-120b-a12b-lora-256k"
    trainer_image = "miles-v0.1.1-fdd808c"
    inference_image = "sglang-v0.5.23-c4182b2"
    max_context_length = 262144
    trainer_gpu = "H200"
    trainer_nodes = 2
    trainer_gpus_per_node = 8
    trainer_max_clients_per_instance = 2
    overrides = {
        # Two slots leave optimizer-state headroom for padded 256K batches.
        "miles_cfg.max_lora_slots": 2,
        "miles_cfg.max_tokens_per_gpu": 32768,
        "miles_cfg.context_parallel_size": 8,
        "miles_cfg.expert_model_parallel_size": 16,
        "miles_cfg.align_sequences_to_parallel_layout": True,
        "sglang_cfg.max_loras_per_batch": 2,
    }


config = Config()
