from spindle.configs.gpt_oss_20b_lora_32k import Config as Parent


class Config(Parent):
    name = "prod-gpt-oss-20b-lora-32k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.requires_proxy_auth": True,
        "platform.modal.environment": "spindle",
        "platform.modal.exclude_reserved_capacity": True,
        "trainer_max_instances": 6,
        "inference_gpus_per_node": 1,
        "inference_min_replicas": 4,
        "inference_max_replicas": 16,
        "sglang_cfg.tp_size": 1,
        "sglang_cfg.max_running_requests": 128,
        "sglang_cfg.max_queued_requests": 4096,
    }


config = Config()
