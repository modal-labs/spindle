from spindle.configs.nemotron3_super_120b_a12b_lora_64k import Config as Parent


class Config(Parent):
    name = "nemotron35-super-vl-120b-a12b-lora-64k"
    trainer_image = "miles-v0.2.0-1dc972b"
    inference_image = "sglang-v0.5.23-c4182b2"
    # Text only: adapters apply to the language model and image inputs are rejected.
    model = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
    overrides = {
        # SGLang sizes Mamba cache snapshots from the top-level config, which lacks the nested chunk size.
        "sglang_cfg.json_model_override_args": '{"mamba_chunk_size": 128}',
    }


config = Config()
