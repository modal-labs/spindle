from spindle.configs.qwen35_9b_instruct_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-instruct-lora-32k-dflash-b200"
    weights_revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    max_context_length = 32768
    trainer_gpu = "B200"
    trainer_gpus_per_node = 1
    trainer_cpu = 32
    trainer_memory_mib = 196608
    trainer_max_clients_per_instance = 1
    inference_gpu = "B200"
    inference_cpu = 24
    inference_memory_mib = 131072
    inference_target_concurrency = 64
    inference_draft_model = "modal-labs/Qwen3.5-9B-DFlash"
    inference_draft_model_revision = "28407c638832073e50f1aacd89e1d15a288b7b1c"
    overrides = {
        "trainer_env.TORCH_COMPILE_DISABLE": "1",
        "miles_cfg.tensor_model_parallel_size": 1,
        "miles_cfg.max_lora_slots": 1,
        "miles_cfg.max_tokens_per_gpu": 18432,
        "sglang_cfg.max_running_requests": 64,
        "sglang_cfg.max_queued_requests": 128,
        "sglang_cfg.max_loaded_loras": 32,
        "sglang_cfg.max_loras_per_batch": 2,
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
    }


config = Config()
