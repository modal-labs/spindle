from spindle.configs.qwen38_27b_lora_64k import Config as Parent


class Config(Parent):
    name = "cv-qwen38-27b-lora-64k"
    trainer_gpu = "B300"
    inference_gpu = "B300"
    overrides = {
        "platform.frontend": "cv",
        "platform.url_label": "cv",
        "platform.requires_proxy_auth": True,
        "platform.modal.environment": "capacity-validation",
    }


config = Config()
