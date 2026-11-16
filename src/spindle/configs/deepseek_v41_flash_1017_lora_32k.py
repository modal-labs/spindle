from spindle.configs.deepseek_v41_flash_1017_lora_8k import Config as Weights
from spindle.configs.deepseek_v41_flash_lora_32k import Config as Parent


class Config(Parent):
    """DeepSeek-V4.1-Flash-1017 LoRA with 32K context."""

    name = "deepseek-v41-flash-1017-lora-32k"
    trainer_image = "miles-v0.1.2"
    inference_image = "spindle.providers.modal.deepseek_v41_image:inference_image"
    model = Weights.model
    weights_revision = Weights.weights_revision


config = Config()
