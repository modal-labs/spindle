from spindle.configs.nemotron35_lightning_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    name = "nemotron35-lightning-30b-a3b-lora-64k-mtp"
    overrides = {
        **Parent.overrides,
        "sglang_cfg.speculative_algorithm": "NEXTN",
        "sglang_cfg.speculative_num_steps": 3,
        "sglang_cfg.speculative_eagle_topk": 1,
        "sglang_cfg.speculative_num_draft_tokens": 4,
        # MTP keeps a Mamba state per draft token. Storing the SSM state in
        # float16 halves that memory, so 640 cached states fit and the sampler
        # runs 128 requests at a time. The trainer keeps its state in float32.
        "sglang_cfg.mamba_ssm_dtype": "float16",
        "sglang_cfg.max_mamba_cache_size": 640,
        "sglang_cfg.mem_fraction_static": 0.9,
    }


config = Config()
