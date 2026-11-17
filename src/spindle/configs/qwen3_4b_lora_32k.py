from spindle.configs._qwen3_lora import Qwen3Lora32k


class Config(Qwen3Lora32k):
    name = "qwen3-4b-lora-32k"
    trainer_image = "miles-v0.1.1-fdd808c"
    inference_image = "sglang-v0.5.23-c4182b2"
    model = "Qwen/Qwen3-4B"
    # The 4B model fits one GPU, so the four trainer GPUs each hold a replica.
    inference_gpus_per_node = 1
    overrides = {
        "miles_cfg.model_type": "qwen3-4B",
        "miles_cfg.tensor_model_parallel_size": 1,
        "sglang_cfg.tp_size": 1,
    }


config = Config()
