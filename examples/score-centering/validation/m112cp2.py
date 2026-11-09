from m111 import Config as Parent


class Config(Parent):
    name = "m112cp2"
    overrides = {"trainer_image": "miles-v0.1.2", "miles_cfg.context_parallel_size": 2}


config = Config()
