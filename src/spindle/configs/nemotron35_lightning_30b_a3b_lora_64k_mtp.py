from spindle.configs.nemotron35_lightning_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    name = "nemotron35-lightning-30b-a3b-lora-64k-mtp"
    overrides = {
        **Parent.overrides,
        "sglang_cfg.speculative_algorithm": "NEXTN",
        "sglang_cfg.speculative_num_steps": 3,
        "sglang_cfg.speculative_eagle_topk": 1,
        "sglang_cfg.speculative_num_draft_tokens": 4,
        # MTP keeps a Mamba state per draft token. With 448 cached states the
        # sampler runs 89 requests at a time and still has room for the KV cache.
        "sglang_cfg.max_mamba_cache_size": 448,
        "sglang_cfg.mem_fraction_static": 0.9,
    }


config = Config()
