from spindle.configs.nemotron35_super_vl_120b_a12b_lora_64k import Config as Parent


class Config(Parent):
    name = "prod-nemotron35-super-vl-120b-a12b-lora-64k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.requires_proxy_auth": True,
        "platform.modal.environment": "spindle",
        "platform.modal.exclude_reserved_capacity": True,
        "trainer_max_instances": 4,
    }


config = Config()
