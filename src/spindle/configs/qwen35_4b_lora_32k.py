from spindle.configs._qwen35_lora_32k import Qwen35Lora32k as Parent


class Config(Parent):
    name = "qwen35-4b-lora-32k"
    model = "Qwen/Qwen3.5-4B"
    trainer_gpu = "B200"
    inference_gpu = "B200"
    inference_target_concurrency = 64
    overrides = {
        **Parent.overrides,
        "sglang_cfg.tp_size": 1,
        "sglang_cfg.attention_backend": "trtllm_mha",
        "sglang_cfg.max_running_requests": 64,
        "sglang_cfg.max_queued_requests": 128,
        "sglang_cfg.max_loaded_loras": 4,
        "sglang_cfg.max_loras_per_batch": 4,
        "sglang_cfg.lora_backend": "csgmv",
        "sglang_cfg.cuda_graph_max_bs_decode": 64,
        "sglang_cfg.speculative_algorithm": "NEXTN",
        "sglang_cfg.speculative_num_steps": 3,
        "sglang_cfg.speculative_eagle_topk": 1,
        "sglang_cfg.speculative_num_draft_tokens": 4,
    }


config = Config()
