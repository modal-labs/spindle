from spindle.configs.qwen35_9b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-lora-16k-single"
    trainer_image = "miles-v0.2.0-82e2671"
    inference_image = "sglang-v0.5.23-e003cf9"
    overrides = {
        "miles_cfg.megatron_to_hf_mode": "raw",
        "miles_cfg.lora_type": "canonical_lora",
        "trainer_max_clients_per_instance": 1,
    }


config = Config()
