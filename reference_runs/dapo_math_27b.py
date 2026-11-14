"""DAPO-Math on the existing two-node Qwen3.8-27B recipe."""

from dapo_math import Config, main


if __name__ == "__main__":
    main(
        config=Config(
            model="Qwen/Qwen3.8-27B",
            tokenizer_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        ),
        recipe="qwen38-27b-lora-256k",
        hardware={
            "trainer": {"gpu": "H200", "nodes": 2, "gpus_per_node": 8},
            "sampler": {
                "gpu": ["H200", "B300", "B200"],
                "gpus_per_replica": 4,
                "replicas": 2,
            },
        },
    )
