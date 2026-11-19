from spindle.configs.deepseek_v41_flash_1017_lora_8k import Config as Weights
from spindle.configs.deepseek_v41_flash_lora_64k import Config as Parent


class Config(Parent):
    """DeepSeek-V4.1-Flash-1017 LoRA with 64K context."""

    name = "deepseek-v41-flash-1017-lora-64k"
    trainer_image = "miles-v0.1.2-99e1ebe"
    inference_image = "sglang-v0.5.23-c4182b2"
    model = Weights.model
    weights_revision = Weights.weights_revision


config = Config()
