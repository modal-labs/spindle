from spindle.configs.qwen35_4b_fft_64k import Config as Parent


class Config(Parent):
    name = "qwen35-4b-fft-64k-h200"
    trainer_gpu = "H200"
    inference_gpu = "H200"


config = Config()
