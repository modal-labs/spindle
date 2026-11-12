"""Deployment settings used for the recorded Miles v0.2.0 run."""

from spindle.configs.qwen35_4b_lora_32k import Config as Parent


class Config(Parent):
    name = "qwen35-4b-miles-v020-validation"
    trainer_image = "miles-v0.2.0"
    trainer_gpu = "H100"
    inference_gpu = "H100"
    session_idle_timeout_s = 300
    pool_idle_timeout_s = 300
    sweep_interval_s = 300


config = Config()
