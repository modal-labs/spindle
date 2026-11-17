from spindle.configs._qwen3_lora import Qwen3Lora32k


class Config(Qwen3Lora32k):
    name = "qwen3-8b-lora-32k"
    model = "Qwen/Qwen3-8B"
    inference_gpu = "B300"
    inference_gpus_per_node = 1
    inference_target_concurrency = 128
    inference_draft_model = "z-lab/Qwen3-8B-DFlash-b16"
    inference_draft_model_revision = "9b41424b7109f9c5413454f481b09a82b85333f4"
    overrides = {
        "sglang_cfg.tp_size": 1,
        "sglang_cfg.dtype": "bfloat16",
        "sglang_cfg.mem_fraction_static": 0.85,
        "sglang_cfg.attention_backend": "flashinfer",
        "sglang_cfg.max_running_requests": 128,
        "sglang_cfg.max_loaded_loras": 4,
        "sglang_cfg.max_loras_per_batch": 4,
        "sglang_cfg.lora_backend": "csgmv",
        "sglang_cfg.cuda_graph_max_bs_decode": 128,
        "sglang_cfg.speculative_algorithm": "DFLASH",
        "sglang_cfg.speculative_dflash_block_size": 16,
    }


config = Config()
