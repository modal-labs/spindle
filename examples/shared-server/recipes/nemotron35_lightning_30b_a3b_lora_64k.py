from spindle.configs.nemotron35_lightning_30b_a3b_lora_64k_h200 import Config as Parent


class Config(Parent):
    name = "prod-nemotron35-lightning-30b-a3b-lora-64k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.requires_proxy_auth": True,
        "platform.modal.environment": "spindle",
        "platform.modal.exclude_reserved_capacity": True,
        "trainer_max_instances": 4,
    }


config = Config()
