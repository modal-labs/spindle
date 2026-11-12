"""Base image provenance and deployment pins; see docs/runtime-images.md."""

IMAGES = {
    "miles-v0.1.0": {
        "upstream": "radixark/miles:v0.1.0@sha256:cd40db923225c4146e90fdf4aa04bc000b71c1e980cc42df6f368de7545eaa09",
        "image": "radixark/miles:v0.1.0@sha256:cd40db923225c4146e90fdf4aa04bc000b71c1e980cc42df6f368de7545eaa09",
    },
    "miles-v0.1.1": {
        "upstream": "radixark/miles:v0.1.1@sha256:6355834f16bacd35d5d40c43f142e3758376f7b2e8d678bccfe870c092bd96bf",
        "image": "radixark/miles:v0.1.1@sha256:6355834f16bacd35d5d40c43f142e3758376f7b2e8d678bccfe870c092bd96bf",
    },
    "miles-v0.1.2": {
        "upstream": "radixark/miles:v0.1.2@sha256:c81d8ba7b91571aac8e0bb2be372c6a9664e1ff7b88f518a8891861bae83e3b7",
        "image": "radixark/miles:v0.1.2@sha256:c81d8ba7b91571aac8e0bb2be372c6a9664e1ff7b88f518a8891861bae83e3b7",
    },
    "sglang-v0.5.21": {
        "upstream": "lmsysorg/sglang:v0.5.21",
        "image": "lmsysorg/sglang:v0.5.21@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469",
    },
    "sglang-v0.5.23": {
        "upstream": "lmsysorg/sglang:v0.5.23",
        "image": "lmsysorg/sglang:v0.5.23@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f",
    },
}

BASE_IMAGES = {name: entry["image"] for name, entry in IMAGES.items()}
