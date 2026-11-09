from spindle.configs.qwen35_35b_a3b_lora_8k import Config as Parent


class Config(Parent):
    name = "qwen35-35b-a3b-mxfp8-lora-8k"
    inference_env = {"SGLANG_FP8_IGNORED_LAYERS": "in_proj_ba"}
    overrides = {
        "miles_cfg.base_quantization": "mxfp8",
        "sglang_cfg.quantization": "mxfp8",
    }


config = Config()
