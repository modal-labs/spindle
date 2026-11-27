from __future__ import annotations

import base64
import math

import numpy as np
import pytest

from spindle.stability import (
    CapturePlan,
    CapturedSequence,
    RecipeStabilityConfig,
    StabilityInfo,
    extract_captured_sequence,
    needed_chunk_keys,
    pack_bits_u16,
    pack_capture,
    resolve_capture_plan,
    unpack_bits_u16,
    unpack_capture,
)


@pytest.mark.parametrize(
    ("recipe_policy", "client_choice", "expected"),
    [
        ("off", None, False),
        ("off", False, False),
        ("allowed", None, False),
        ("allowed", False, False),
        ("allowed", True, True),
        ("default", None, True),
        ("default", True, True),
        ("default", False, False),
    ],
)
def test_policy_resolution_matrix(
    recipe_policy: str,
    client_choice: bool | None,
    expected: bool,
) -> None:
    recipe = RecipeStabilityConfig(routed_experts=recipe_policy)  # type: ignore[arg-type]
    plan = resolve_capture_plan(StabilityInfo(routed_experts=client_choice), recipe)
    assert plan is not None
    assert plan.routed_experts is expected
    assert plan.sampler_draws is True


def test_policy_off_rejects_explicit_client_request() -> None:
    recipe = RecipeStabilityConfig(routed_experts="off")
    with pytest.raises(ValueError, match=r"routed_experts.*disabled"):
        resolve_capture_plan({"routed_experts": True}, recipe)


def test_sampling_probabilities_implies_mask_and_rejects_contradictions() -> None:
    recipe = RecipeStabilityConfig(
        sampling_mask="allowed",
        sampling_probabilities="allowed",
        sampling_mask_top_k=64,
    )
    plan = resolve_capture_plan({"sampling_probabilities": True}, recipe)
    assert plan is not None
    assert plan.sampling_probabilities is True
    assert plan.sampling_mask is True
    assert plan.sampling_mask_top_k == 64

    with pytest.raises(ValueError, match="requires sampling_mask"):
        resolve_capture_plan(
            {"sampling_probabilities": True, "sampling_mask": False}, recipe
        )

    off_mask = RecipeStabilityConfig(
        sampling_mask="off", sampling_probabilities="allowed"
    )
    with pytest.raises(ValueError, match=r"sampling_mask.*disabled"):
        resolve_capture_plan({"sampling_probabilities": True}, off_mask)


def test_false_or_none_disables_capture() -> None:
    recipe = RecipeStabilityConfig(sampling_mask="default")
    assert resolve_capture_plan(False, recipe) is None
    assert resolve_capture_plan(None, recipe) is None
    plan = resolve_capture_plan(True, recipe)
    assert plan is not None
    assert plan.sampling_mask is True


@pytest.mark.parametrize("bits", [9, 10, 12, 15])
@pytest.mark.parametrize("count", [0, 1, 7, 8, 9, 1023, 4096])
def test_bitpack_u16_roundtrip(bits: int, count: int) -> None:
    rng = np.random.default_rng(bits * 1000 + count)
    values = rng.integers(0, 1 << bits, size=count, dtype=np.uint16)
    packed = pack_bits_u16(values, bits)
    expected_bytes = ((count + 7) // 8) * bits if count else 0
    assert len(packed) == expected_bytes
    restored = unpack_bits_u16(packed, count, bits)
    np.testing.assert_array_equal(restored, values.astype(np.int32))


@pytest.mark.parametrize("compression", ["none", "zstd"])
@pytest.mark.parametrize("max_expert", [127, 287, -1])
def test_pack_unpack_chunked_and_selective_fields(
    compression: str, max_expert: int
) -> None:
    rng = np.random.default_rng(42)
    prompt = np.arange(10, 26, dtype=np.int64)
    response = np.arange(100, 132, dtype=np.int64)
    n_input = prompt.size + response.size - 1
    if max_expert < 0:
        routes = np.full((n_input, 4, 8), -1, dtype=np.int32)
        routes[:10] = rng.integers(0, 64, size=(10, 4, 8), dtype=np.int32)
    else:
        routes = rng.integers(0, max_expert + 1, size=(n_input, 4, 8), dtype=np.int32)
        routes[0, 0, 0] = max_expert

    mask_ids = []
    offsets = [0]
    mask_lps = []
    chosen_lps = []
    for tok in response:
        support = [int(tok), int(tok) + 1, int(tok) + 2, int(tok) + 3]
        probs = [0.4, 0.3, 0.2, 0.1]
        lps = [math.log(p) for p in probs]
        mask_ids.extend(support)
        offsets.append(len(mask_ids))
        mask_lps.extend(lps)
        chosen_lps.append(lps[0])

    draws = rng.integers(0, 50_000, size=(response.size, 8), dtype=np.int64)
    seq = CapturedSequence(
        sequence_id="req-1:0",
        session_id="sess-1",
        created_at=1234.5,
        prompt_tokens=prompt,
        response_tokens=response,
        temperature=0.8,
        routed_experts=routes,
        sampling_mask_ids=np.asarray(mask_ids, dtype=np.int64),
        sampling_mask_offsets=np.asarray(offsets, dtype=np.int64),
        sampling_mask_logprobs=np.asarray(mask_lps, dtype=np.float32),
        sampling_logprobs=np.asarray(chosen_lps, dtype=np.float32),
        sampler_draws=draws,
    )

    packed = pack_capture(
        seq,
        target_chunk_bytes=1024,
        compression=compression,  # type: ignore[arg-type]
    )
    assert len(packed.chunks) >= 4
    expected_encoding = (
        "int32" if max_expert < 0 else "uint8" if max_expert <= 255 else "bitpack"
    )
    assert packed.header["routes_meta"]["encoding"] == expected_encoding

    chunk_map = dict(packed.chunks)
    full = unpack_capture(packed.header, chunk_map)
    np.testing.assert_array_equal(full.prompt_tokens, prompt)
    np.testing.assert_array_equal(full.response_tokens, response)
    np.testing.assert_array_equal(full.routed_experts, routes)  # type: ignore[arg-type]
    np.testing.assert_array_equal(full.sampling_mask_ids, seq.sampling_mask_ids)  # type: ignore[arg-type]
    np.testing.assert_array_equal(
        full.sampling_mask_offsets,
        seq.sampling_mask_offsets,  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(
        full.sampling_mask_logprobs,
        seq.sampling_mask_logprobs,
        rtol=1e-6,  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(
        full.sampling_logprobs,
        seq.sampling_logprobs,
        rtol=1e-6,  # type: ignore[arg-type]
    )
    np.testing.assert_array_equal(full.sampler_draws, draws)  # type: ignore[arg-type]

    # Selective read: requesting only sampler_draws should not require route/mask chunks.
    draw_keys = needed_chunk_keys(packed.header, {"sampler_draws"})
    assert all(k.startswith("draws:") for k in draw_keys)
    partial = unpack_capture(
        packed.header,
        {k: chunk_map[k] for k in draw_keys},
        fields={"sampler_draws"},
    )
    assert partial.routed_experts is None
    assert partial.sampling_mask_ids is None
    np.testing.assert_array_equal(partial.sampler_draws, draws)  # type: ignore[arg-type]
    assert "routed_experts" in partial.recorded_fields


def test_extract_captured_sequence_strips_raw_sglang_meta() -> None:
    prompt = [1, 2, 3]
    response = [10, 11]
    routes = np.arange(4 * 2 * 4, dtype="<i2").reshape(4, 2, 4)
    meta = {
        "routed_experts": base64.b64encode(routes.tobytes()).decode("ascii"),
        "routed_experts_dtype": "int16",
        "output_token_sampling_mask": [[10, 20], [11, 21]],
        "output_token_sampling_logprobs": [
            [math.log(0.7), math.log(0.3)],
            [math.log(0.6), math.log(0.4)],
        ],
        "output_token_alt_tokens": [[10, 20, 10, 20], [11, 21, 11, 21]],
        "cached_tokens": 2,
    }
    plan = CapturePlan(
        routed_experts=True,
        sampling_mask=True,
        sampling_probabilities=True,
        sampler_draws=True,
    )
    captured = extract_captured_sequence(
        meta,
        prompt,
        response,
        session_id="s-1",
        sequence_id="req:0",
        temperature=0.7,
        plan=plan,
        num_layers=2,
        experts_per_token=4,
        strip_meta=True,
    )
    assert list(meta.keys()) == ["cached_tokens"]
    assert captured.routed_experts is not None
    assert captured.routed_experts.shape == (4, 2, 4)
    np.testing.assert_array_equal(captured.routed_experts, routes.astype(np.int32))
    assert captured.sampling_mask_logprobs is not None
    assert captured.sampler_draws is not None
    assert captured.sampler_draws.shape == (2, 4)
