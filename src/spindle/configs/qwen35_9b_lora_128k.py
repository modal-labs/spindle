from spindle.configs._qwen35_9b_instruct_lora import Qwen35InstructLora as Parent


class Config(Parent):
    name = "qwen35-9b-lora-128k"
    trainer_image = "miles-v0.2.0-3f62994"
    inference_image = "sglang-v0.5.23-e003cf9"
    max_context_length = 131_072
    trainer_gpu = "H200"
    inference_gpu = "H200"
    trainer_max_clients_per_instance = 8
    inference_min_replicas = 8
    inference_max_replicas = 8
    overrides = {
        "miles_cfg.megatron_to_hf_mode": "raw",
        "miles_cfg.lora_type": "canonical_lora",
        "miles_cfg.max_tokens_per_gpu": 131_072,
        "miles_cfg.max_lora_slots": 8,
        "sglang_cfg.max_running_requests": 32,
        "sglang_cfg.max_queued_requests": 8,
    }


config = Config()
