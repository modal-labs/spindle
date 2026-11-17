from spindle.configs.nemotron3_super_120b_a12b_lora_64k import Config as Parent


class Config(Parent):
    name = "nemotron35-super-vl-120b-a12b-lora-64k"
    trainer_image = "miles-v0.1.1-fdd808c"
    inference_image = "sglang-v0.5.23-c4182b2"
    # Text only: adapters train the language model; the vision encoder is served frozen.
    model = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"


config = Config()
