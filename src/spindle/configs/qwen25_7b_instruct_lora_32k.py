from spindle.configs._qwen3_lora import Qwen3Lora32k


class Config(Qwen3Lora32k):
    name = "qwen25-7b-instruct-lora-32k"
    model = "Qwen/Qwen2.5-7B-Instruct"
    inference_gpus_per_node = 1
    inference_max_replicas = 16
    overrides = {
        "miles_cfg.model_type": "qwen2.5-7B",
        "sglang_cfg.tp_size": 1,
    }


config = Config()
