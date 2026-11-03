from spindle.configs.nemotron35_lightning_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    name = "cv-nemotron35-lightning-30b-a3b-lora-64k"
    trainer_gpu = "B300"
    inference_gpu = "B300"
    overrides = {
        "platform.frontend": "cv",
        "platform.url_label": "cv",
        "platform.requires_proxy_auth": True,
        "platform.modal.environment": "capacity-validation",
    }


config = Config()
