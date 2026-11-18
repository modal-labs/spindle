from spindle.configs._qwen35_9b_instruct_lora import Qwen35InstructLora as Parent


class Config(Parent):
    name = "qwen35-9b-instruct-lora-16k-dp2"
    trainer_image = "miles-v0.2.0-ac54685"
    inference_image = "sglang-v0.5.23-e003cf9"
    overrides = {
        "miles_cfg.megatron_to_hf_mode": "raw",
        "miles_cfg.lora_type": "canonical_lora",
        "miles_cfg.tensor_model_parallel_size": 4,
    }


config = Config()
