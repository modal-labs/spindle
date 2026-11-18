from spindle.configs._qwen35_9b_instruct_lora import Qwen35InstructLora as Parent


class Config(Parent):
    name = "qwen35-9b-instruct-lora-16k"
    trainer_image = "miles-v0.2.0-e1b172f"
    inference_image = "sglang-v0.5.23-e003cf9"
    trainer_gpu = "B200"
    trainer_gpus_per_node = 1
    inference_gpu = "B200"
    inference_draft_model = "modal-labs/Qwen3.5-9B-DFlash"
    inference_draft_model_revision = "28407c638832073e50f1aacd89e1d15a288b7b1c"
    overrides = {
        "miles_cfg.megatron_to_hf_mode": "raw",
        "miles_cfg.lora_type": "canonical_lora",
        "miles_cfg.tensor_model_parallel_size": 1,
        "miles_cfg.cli_options": {},
        "sglang_cfg.max_running_requests": 64,
        "sglang_cfg.max_mamba_cache_size": 384,
        "sglang_cfg.cuda_graph_max_bs_decode": 64,
        "sglang_cfg.attention_backend": "trtllm_mha",
        "sglang_cfg.linear_attn_prefill_backend": "triton",
        "sglang_cfg.linear_attn_decode_backend": "triton",
        "sglang_cfg.linear_attn_verify_backend": "triton",
        "sglang_cfg.mamba_ssm_dtype": "float32",
        "sglang_cfg.mamba_radix_cache_strategy": "extra_buffer",
        "sglang_cfg.speculative_algorithm": "DFLASH",
        "sglang_cfg.speculative_dflash_block_size": 8,
        "sglang_cfg.speculative_draft_attention_backend": "fa4",
        "sglang_cfg.speculative_draft_model_quantization": "unquant",
        "sglang_cfg.speculative_draft_kv_cache_dtype": "bfloat16",
    }


config = Config()
