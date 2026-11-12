from spindle.configs.qwen38_27b_lora_16k_b200 import Config as Parent


class Config(Parent):
    name = "qwen38-27b-lora-32k-dflash-b200"
    max_context_length = 32768
    inference_gpus_per_node = 2
    inference_cpu = 64
    inference_memory_mib = 262144
    inference_draft_model = "incoai/Qwen3.8-27B-DFlash2"
    inference_draft_model_revision = "015e795645c74b1a0eeef3b570031fb62e769bc5"
    overrides = {
        **Parent.overrides,
        "miles_cfg.max_tokens_per_gpu": 32768,
        "sglang_cfg.tp_size": 2,
        "sglang_cfg.dtype": "bfloat16",
        "sglang_cfg.mem_fraction_static": 0.85,
        "sglang_cfg.max_running_requests": 64,
        "sglang_cfg.max_mamba_cache_size": 384,
        "sglang_cfg.max_queued_requests": 128,
        "sglang_cfg.max_loaded_loras": 4,
        "sglang_cfg.max_loras_per_batch": 2,
        "sglang_cfg.lora_backend": "csgmv",
        "sglang_cfg.schedule_policy": "fcfs",
        "sglang_cfg.cuda_graph_max_bs_decode": 64,
        "sglang_cfg.chunked_prefill_size": 8192,
        "sglang_cfg.disable_prefill_cuda_graph": True,
        "sglang_cfg.language_only": True,
        "sglang_cfg.linear_attn_prefill_backend": "triton",
        "sglang_cfg.linear_attn_decode_backend": "triton",
        "sglang_cfg.linear_attn_verify_backend": "triton",
        "sglang_cfg.mamba_ssm_dtype": "float32",
        "sglang_cfg.mamba_radix_cache_strategy": "extra_buffer",
        "sglang_cfg.kv_cache_dtype": "bfloat16",
        "sglang_cfg.speculative_algorithm": "DFLASH",
        "sglang_cfg.speculative_dflash_block_size": 8,
        "sglang_cfg.speculative_draft_attention_backend": "fa4",
        "sglang_cfg.speculative_draft_model_quantization": "unquant",
        "sglang_cfg.speculative_draft_kv_cache_dtype": "bfloat16",
    }


config = Config()
