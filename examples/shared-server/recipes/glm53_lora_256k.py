from spindle.configs.glm53_lora_256k import Config as Parent


class Config(Parent):
    name = "prod-glm53-lora-256k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.requires_proxy_auth": True,
        "platform.modal.environment": "spindle",
        "platform.modal.exclude_reserved_capacity": True,
        "trainer_max_instances": 4,
    }


config = Config()
