from spindle.configs.nemotron3_super_120b_a12b_lora_64k_b300 import Config as Parent


class Config(Parent):
    name = "cv-nemotron3-super-120b-a12b-lora-64k"
    overrides = {
        "platform.frontend": "cv",
        "platform.url_label": "cv",
        "platform.requires_proxy_auth": True,
        "platform.modal.environment": "capacity-validation",
    }


config = Config()
