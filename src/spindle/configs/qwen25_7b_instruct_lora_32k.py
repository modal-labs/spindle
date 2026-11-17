from spindle.configs._qwen3_lora import Qwen3Lora32k


class Config(Qwen3Lora32k):
    name = "qwen25-7b-instruct-lora-32k"
    trainer_image = "miles-v0.1.1-fdd808c"
    inference_image = "sglang-v0.5.23-c4182b2"
    model = "Qwen/Qwen2.5-7B-Instruct"
    inference_gpus_per_node = 1
    inference_max_replicas = 16
    overrides = {
        "miles_cfg.model_type": "qwen2.5-7B",
        "sglang_cfg.tp_size": 1,
    }


config = Config()
