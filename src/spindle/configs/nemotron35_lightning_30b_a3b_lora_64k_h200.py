from spindle.configs.nemotron35_lightning_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    name = "nemotron35-lightning-30b-a3b-lora-64k-h200"
    trainer_image = "miles-v0.1.1-52a7a99"
    inference_image = "sglang-v0.5.23-c4182b2"
    trainer_gpu = "H200"
    inference_gpu = "H200"
    inference_target_concurrency = 89
    overrides = {
        "miles_cfg.tensor_model_parallel_size": 2,
        "miles_cfg.expert_model_parallel_size": 2,
        "miles_cfg.cli_options.recompute_method": "uniform",
        "miles_cfg.cli_options.recompute_num_layers": 1,
        "sglang_cfg.max_mamba_cache_size": 448,
        "sglang_cfg.cuda_graph_backend_prefill": None,
    }


config = Config()
