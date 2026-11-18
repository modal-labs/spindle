"""Experimental opt-in GLM Flash shared-prefix LoRA; one trainer and adapter."""

from spindle.configs.glm53_flash_lora_64k import Config as Parent


class Config(Parent):
    name = "glm53-flash-lora-shared-prefix-b300"
    trainer_image = "spindle.providers.modal.shared_prefix_image:image"
    inference_image = "sglang-v0.5.23-c4182b2-glm53"
    trainer_gpu = "B300"
    trainer_gpus_per_node = 8
    trainer_nodes = 1
    trainer_memory_mib = 786432
    trainer_cpu = 64
    trainer_max_clients_per_instance = 1
    trainer_env = {
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }
    inference_gpu = "B300"
    inference_memory_mib = 786432
    max_context_length = 32768
    miles_cfg = {
        "shared_prefix_training": True,
        "model_type": "glm5.3-flash",
        "megatron_to_hf_mode": "raw",
        "lora_type": "canonical_lora",
        "tensor_model_parallel_size": 2,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
        "max_lora_slots": 1,
        "max_lora_rank": 32,
        "max_tokens_per_gpu": 49152,
        "target_modules": [
            "model.language_model.layers.*.q_proj",
            "model.language_model.layers.*.k_proj",
            "model.language_model.layers.*.v_proj",
            "model.language_model.layers.*.b_proj",
            "model.language_model.layers.*.f_a_proj",
            "model.language_model.layers.*.f_b_proj",
            "model.language_model.layers.*.g_a_proj",
            "model.language_model.layers.*.g_b_proj",
            "model.language_model.layers.*.o_proj",
            "model.language_model.layers.*.q_a_proj",
            "model.language_model.layers.*.q_b_proj",
            "model.language_model.layers.*.kv_a_proj_with_mqa",
            "model.language_model.layers.*.gate_proj",
            "model.language_model.layers.*.up_proj",
            "model.language_model.layers.*.gate_up_proj",
            "model.language_model.layers.*.down_proj",
        ],
        "cli_options": {
            "lora_kernel_policy": "rowwise",
            "deterministic_mode": True,
            "use_rollout_routing_replay": False,
            "enable_mtp_training": False,
            "moe_permute_fusion": False,
            "moe_router_fusion": False,
            "moe_aux_loss_coeff": [0.0],
            "cross_entropy_loss_fusion": False,
            "tp_comm_overlap": False,
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
    }
    overrides = {
        "platform.modal.exclude_reserved_capacity": True,
        "sglang_cfg.enable_return_routed_experts": False,
    }


config = Config()
