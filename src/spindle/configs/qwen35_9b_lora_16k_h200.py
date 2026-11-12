from spindle.configs.qwen35_9b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-lora-16k-h200"
    trainer_gpu = "H200"
    inference_gpu = "H200"


config = Config()
