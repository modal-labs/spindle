from spindle.configuration import BaseConfig


class Config(BaseConfig):
    """Kimi K3 LoRA with 32K context on the MXFP4 release weights.

    Adapters train attention, the dense first layer and the shared experts.
    Routed experts stay frozen in MXFP4 on both the trainer and the sampler.
    """

    name = "kimi-k3-lora-32k"
    model = "moonshotai/Kimi-K3"
    weights_revision = "f831ab66814297da540d832a5235f8e904f29d06"
    max_context_length = 32768
    backend = "miles"
    trainer_image = "spindle.providers.modal.kimi_k3_image:trainer_image"
    trainer_gpu = "B300"
    trainer_gpus_per_node = 8
    trainer_nodes = 2
    trainer_cpu = 64
    trainer_memory_mib = 1_200_000
    trainer_disk_mib = 600_000
    trainer_timeout_s = 172800
    trainer_max_clients_per_instance = 2
    miles_cfg = {
        "model_type": "kimi-k3",
        "megatron_to_hf_mode": "raw",
        "tensor_model_parallel_size": 8,
        "expert_model_parallel_size": 16,
        "expert_tensor_parallel_size": 1,
        "target_modules": [
            "language_model.model.layers.*.self_attn.q_a_proj",
            "language_model.model.layers.*.self_attn.kv_a_proj_with_mqa",
            "language_model.model.layers.*.self_attn.o_proj",
            "language_model.model.layers.*.mlp.gate_proj",
            "language_model.model.layers.*.mlp.up_proj",
            "language_model.model.layers.*.mlp.down_proj",
            "language_model.model.layers.*.block_sparse_moe.shared_experts.gate_proj",
            "language_model.model.layers.*.block_sparse_moe.shared_experts.up_proj",
            "language_model.model.layers.*.block_sparse_moe.shared_experts.down_proj",
        ],
        "max_tokens_per_gpu": 32768,
        "max_lora_slots": 2,
        "max_lora_rank": 32,
        "default_lora_alpha": 32,
        "cli_options": {
            "model_name": "kimi_k3",
            # The adapter layout Kimi K3 LoRA requires, even with routed experts frozen.
            "experts_shared_outer_loras": True,
            "attention_backend": "auto",
            "qkv_format": "thd",
            # Multi-LoRA shared-expert collectives must not race expert dispatch.
            "moe_shared_expert_overlap": False,
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
            "distributed_timeout_minutes": 120,
        },
    }
    trainer_env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        # Trainer nodes connect over RDMA, not multi-node NVLink.
        "NCCL_MNNVL_ENABLE": "0",
        "NCCL_NVLS_ENABLE": "0",
    }
    inference_image = "spindle.providers.modal.kimi_k3_image:inference_image"
    # The 1.5 TB MXFP4 checkpoint fits on one node only with B300 memory.
    inference_gpu = "B300"
    inference_gpus_per_node = 8
    inference_cpu = 32
    inference_memory_mib = 1_048_576
    inference_disk_mib = 600_000
    # A sampler takes over an hour to start, so one starts with the trainer
    # and stays up until the run ends.
    inference_min_replicas = 1
    inference_max_replicas = 2
    inference_startup_timeout_s = 7200
    inference_env = {"SGLANG_NUMA_BIND_V2": "0"}
    sglang_cfg = {
        "tp_size": 8,
        "ep_size": 1,
        "quantization": "compressed-tensors",
        # Marlin is the MXFP4 MoE runner with a LoRA path.
        "moe_runner_backend": "marlin",
        "lora_backend": "triton",
        "experts_shared_outer_loras": True,
        # Each running request keeps five linear-attention states.
        "max_mamba_cache_size": 160,
        "trust_remote_code": True,
        # Serve text only. The tokenizer stays loaded so clients can stop on strings.
        "enable_multimodal": False,
        "mem_fraction_static": 0.85,
        "max_running_requests": 32,
        "max_queued_requests": 128,
        "cuda_graph_max_bs_decode": 32,
        "max_loaded_loras": 4,
        "max_loras_per_batch": 2,
        "lora_strict_loading": True,
        "model_loader_extra_config": {"enable_multithread_load": True},
        "weight_loader_drop_cache_after_load": True,
    }


config = Config()
