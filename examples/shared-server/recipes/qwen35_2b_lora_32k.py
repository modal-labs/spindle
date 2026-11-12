from spindle.configs.qwen35_2b_lora_32k import Config as Parent


class Config(Parent):
    name = "prod-qwen35-2b-lora-32k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.requires_proxy_auth": True,
        "platform.modal.environment": "spindle",
        "platform.modal.exclude_reserved_capacity": True,
        "trainer_max_instances": 6,
        "inference_min_replicas": 4,
        "inference_max_replicas": 16,
    }


config = Config()
