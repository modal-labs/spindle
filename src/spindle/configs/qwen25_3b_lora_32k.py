from spindle.configs.qwen3_4b_lora_32k import Config as Parent


class Config(Parent):
    name = "qwen25-3b-lora-32k"
    model = "Qwen/Qwen2.5-3B"
    inference_max_replicas = 8
    overrides = {
        "miles_cfg.model_type": "qwen2.5-3B",
    }


config = Config()
