from spindle.configs.deepseek_v41_flash_lora_8k import Config as Parent


class Config(Parent):
    """DeepSeek-V4.1-Flash-1017 LoRA on attention and the shared expert."""

    name = "deepseek-v41-flash-1017-lora-8k"
    trainer_image = "miles-v0.1.2-f2f8447"
    inference_image = "sglang-v0.5.21-a3e0e17"
    model = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    weights_revision = "1bae0e1088993b9b82de08f59de5be5a34e42170"


config = Config()
