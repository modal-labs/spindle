"""Spindle extensions to Tinker SamplingClient builders with duck-typed stability info."""

from __future__ import annotations

import asyncio
import dataclasses
import time

import tinker
from tinker import (
    SamplingClient as TinkerSamplingClient,
    ServiceClient,
    TrainingClient,
    types,
)
from tinker.lib.api_future_impl import _APIFuture
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from tinker.lib.retry_handler import RetryConfig

from .stability import (
    StabilityInfo,
    StabilityRequest,
    parse_stability_request,
    serialize_stability_request,
)


def _unwrap_client(client: object) -> TinkerSamplingClient:
    # tinker >= 0.38 returns BackcompatSamplingClientFuture which subclasses SamplingClient.
    result_fn = getattr(client, "result", None)
    if callable(result_fn):
        return result_fn()
    return client


async def _unwrap_client_async(client: object) -> TinkerSamplingClient:
    result_async_fn = getattr(client, "result_async", None)
    if callable(result_async_fn):
        return await result_async_fn()
    return _unwrap_client(client)


def _attach_sequence_ids(
    response: types.SampleResponse,
    sequence_ids: list[str] | None,
) -> types.SampleResponse:
    if not sequence_ids:
        return response
    if len(sequence_ids) != len(response.sequences):
        raise RuntimeError(
            f"expected {len(response.sequences)} sequence IDs, got {len(sequence_ids)}"
        )
    updated = [
        dataclasses.replace(seq, sequence_id=seq_id)
        for seq, seq_id in zip(response.sequences, sequence_ids, strict=True)
    ]
    return dataclasses.replace(response, sequences=updated)


class StabilitySamplingClient:
    """Wraps a Tinker SamplingClient to send duck-typed record_stability_info."""

    def __init__(
        self,
        client: TinkerSamplingClient,
        *,
        record_stability_info: StabilityRequest = False,
    ) -> None:
        self._client = _unwrap_client(client)
        self._record_stability_info = parse_stability_request(record_stability_info)
        self._wire_stability = serialize_stability_request(record_stability_info)

    def __getattr__(self, name: str) -> object:
        # Guard private/dunder lookups so pickle/copy never recurse before _client exists.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._client, name)

    @property
    def record_stability_info(self) -> StabilityInfo | None:
        return self._record_stability_info

    def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> AwaitableConcurrentFuture[types.SampleResponse]:
        if self._record_stability_info is None:
            return self._client.sample(
                prompt=prompt,
                num_samples=num_samples,
                sampling_params=sampling_params,
                include_prompt_logprobs=include_prompt_logprobs,
                topk_prompt_logprobs=topk_prompt_logprobs,
            )

        client = self._client

        async def _run() -> types.SampleResponse:
            # Tinker >= 0.40 lazily joins cloned sampling clients before allocating seq_id.
            ensure_joined = getattr(client, "_ensure_joined", None)
            if callable(ensure_joined):
                await ensure_joined()
            elif getattr(
                client, "_cloned_sampler_id", None
            ) is not None and not getattr(client, "_sampling_session_id", None):
                join_fn = getattr(client, "_join_cloned_session", None)
                if callable(join_fn):
                    await join_fn()

            seq_id = client._request_id_counter
            client._request_id_counter += 1
            started = time.time()
            request = types.SampleRequest(
                sampling_session_id=client._sampling_session_id,
                seq_id=seq_id,
                num_samples=num_samples,
                prompt=prompt,
                sampling_params=sampling_params,
                prompt_logprobs=include_prompt_logprobs,
                topk_prompt_logprobs=topk_prompt_logprobs,
            )
            body = {
                **request.model_dump(exclude_none=True, mode="json"),
                "record_stability_info": self._wire_stability,
            }
            estimated_bytes = client.holder.estimate_bytes_count_in_model_input(prompt)

            async def _post_once() -> types.UntypedAPIFuture:
                backoff_delay = 0.5
                while True:
                    backoff_until = getattr(
                        client.holder, "_sample_backoff_until", None
                    )
                    if backoff_until is not None and backoff_until > time.time():
                        await asyncio.sleep(backoff_until - time.time())
                    try:
                        async with client.holder.sample_dispatch_rate_limit(
                            estimated_bytes
                        ):
                            with client.holder.aclient(
                                ClientConnectionPoolType.SAMPLE
                            ) as api:
                                return await api.post(
                                    "/api/v1/asample",
                                    body=body,
                                    cast_to=types.UntypedAPIFuture,
                                )
                    except tinker.APIStatusError as exc:
                        if exc.status_code == 429 or exc.status_code >= 500:
                            await asyncio.sleep(backoff_delay)
                            backoff_delay = min(backoff_delay * 2.0, 5.0)
                            continue
                        raise

            retry_handler = getattr(client, "retry_handler", None)
            if retry_handler is not None and callable(
                getattr(retry_handler, "execute", None)
            ):
                future = await retry_handler.execute(_post_once)
            else:
                future = await _post_once()

            response = await _APIFuture(
                types.SampleResponse,
                client.holder,
                future,
                request_start_time=started,
                request_type="Sample",
                queue_state_observer=client,
            ).result_async()
            return _attach_sequence_ids(
                response, getattr(future, "sample_sequence_ids", None)
            )

        return client.holder.run_coroutine_threadsafe(_run())

    async def sample_async(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> types.SampleResponse:
        if self._record_stability_info is None:
            return await self._client.sample_async(
                prompt=prompt,
                num_samples=num_samples,
                sampling_params=sampling_params,
                include_prompt_logprobs=include_prompt_logprobs,
                topk_prompt_logprobs=topk_prompt_logprobs,
            )
        return await self.sample(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
        ).result_async()


def create_sampling_client(
    service: ServiceClient,
    *,
    model_path: str | None = None,
    base_model: str | None = None,
    retry_config: RetryConfig | None = None,
    record_stability_info: StabilityRequest = False,
) -> StabilitySamplingClient:
    """Create a SamplingClient accepting bool | dict | StabilityInfo."""
    parse_stability_request(record_stability_info)
    client = _unwrap_client(
        service.create_sampling_client(
            model_path=model_path,
            base_model=base_model,
            retry_config=retry_config,
        )
    )
    return StabilitySamplingClient(client, record_stability_info=record_stability_info)


async def create_sampling_client_async(
    service: ServiceClient,
    *,
    model_path: str | None = None,
    base_model: str | None = None,
    retry_config: RetryConfig | None = None,
    record_stability_info: StabilityRequest = False,
) -> StabilitySamplingClient:
    """Async variant of create_sampling_client accepting bool | dict | StabilityInfo."""
    parse_stability_request(record_stability_info)
    client = await _unwrap_client_async(
        await service.create_sampling_client_async(
            model_path=model_path,
            base_model=base_model,
            retry_config=retry_config,
        )
    )
    return StabilitySamplingClient(client, record_stability_info=record_stability_info)


def save_weights_and_get_sampling_client(
    training: TrainingClient,
    *,
    name: str | None = None,
    retry_config: RetryConfig | None = None,
    record_stability_info: StabilityRequest = False,
) -> StabilitySamplingClient:
    """Save weights and return a SamplingClient accepting bool | dict | StabilityInfo."""
    parse_stability_request(record_stability_info)
    client = _unwrap_client(
        training.save_weights_and_get_sampling_client(
            name=name,
            retry_config=retry_config,
        )
    )
    return StabilitySamplingClient(client, record_stability_info=record_stability_info)


async def save_weights_and_get_sampling_client_async(
    training: TrainingClient,
    *,
    name: str | None = None,
    retry_config: RetryConfig | None = None,
    record_stability_info: StabilityRequest = False,
) -> StabilitySamplingClient:
    """Async variant of save_weights_and_get_sampling_client."""
    parse_stability_request(record_stability_info)
    client = await _unwrap_client_async(
        await training.save_weights_and_get_sampling_client_async(
            name=name,
            retry_config=retry_config,
        )
    )
    return StabilitySamplingClient(client, record_stability_info=record_stability_info)


__all__ = [
    "StabilityInfo",
    "StabilitySamplingClient",
    "create_sampling_client",
    "create_sampling_client_async",
    "save_weights_and_get_sampling_client",
    "save_weights_and_get_sampling_client_async",
]
