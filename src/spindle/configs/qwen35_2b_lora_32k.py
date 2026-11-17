from spindle.configs._qwen35_lora_32k import Qwen35Lora32k as Parent


class Config(Parent):
    name = "qwen35-2b-lora-32k"
    trainer_image = "miles-v0.1.1-fdd808c"
    inference_image = "sglang-v0.5.23-803bccb"
    model = "Qwen/Qwen3.5-2B"
    # Miles has no 2B preset; keep the shared Qwen3.5 spec and tied embeddings.
    # Linear-attention dimensions are loaded from the checkpoint's text_config.
    overrides = {
        "inference_gpu": "H100",
        "sglang_cfg.max_running_requests": 64,
        "sglang_cfg.max_queued_requests": 8,
        "miles_cfg.cli_options.num_layers": 24,
        "miles_cfg.cli_options.hidden_size": 2048,
        "miles_cfg.cli_options.ffn_hidden_size": 6144,
        "miles_cfg.cli_options.num_attention_heads": 8,
        "miles_cfg.cli_options.num_query_groups": 2,
    }


config = Config()
