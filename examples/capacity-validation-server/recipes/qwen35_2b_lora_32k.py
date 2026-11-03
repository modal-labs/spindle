from spindle.configs.qwen35_2b_lora_32k import Config as Parent


class Config(Parent):
    name = "cv-qwen35-2b-lora-32k"
    trainer_gpu = "B300"
    inference_gpu = "B300"
    overrides = {
        "platform.frontend": "cv",
        "platform.url_label": "cv",
        "platform.requires_proxy_auth": True,
        "platform.modal.environment": "capacity-validation",
    }


config = Config()
