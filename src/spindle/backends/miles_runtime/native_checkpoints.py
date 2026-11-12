"""Use Miles v0.2.0's adapter directory lifecycle without changing its state format."""

from __future__ import annotations

from contextlib import closing
from inspect import signature
from pathlib import Path
import shutil
import asyncio

from .volumes import refresh_native_capture, validate_native_files


class NativeSourceError(ValueError):
    """The source was rejected before any slot state changed."""


def uses_native_checkpoints(bridge) -> bool:
    return (
        "source" in signature(bridge.load_slot).parameters
        and "root" in signature(bridge.save_slot).parameters
    )


class NativeCheckpoints:
    def __init__(self, bridge, args, config):
        # The native checkpoint library is unavailable in legacy Miles images.
        from miles.backends.training_utils.checkpoint.adapter import (
            commit,
            parametrization,
        )
        from miles.utils.hf_utils.config import load_hf_config

        self.bridge = bridge
        self.args = args
        self.commit = commit
        self.read_parametrization = parametrization.read_parametrization
        self.base_fingerprint = parametrization.base_model_fingerprint(
            load_hf_config(config.hf_checkpoint).to_dict()
        )
        self.slots: dict[int, tuple[int, float]] = {}

    def fields(self, rank: int, alpha: float) -> dict:
        return {
            "base_model": self.args.hf_checkpoint,
            "base_model_fingerprint": self.base_fingerprint,
            "lora_rank": rank,
            "lora_alpha": alpha,
            "lora_type": self.args.lora_type,
            "experts_shared_outer_loras": self.args.experts_shared_outer_loras,
        }

    async def load_slot(
        self, slot, rank, alpha, *, checkpoint=None, restore_optimizer=True
    ):
        if checkpoint is not None:
            try:
                saved = self.read_parametrization(Path(checkpoint))
            except (ValueError, TypeError, OSError, AssertionError) as exc:
                raise NativeSourceError(f"invalid native checkpoint: {exc}") from exc
            expected = self.fields(rank, alpha)
            for name in (
                "lora_rank",
                "lora_alpha",
                "lora_type",
                "experts_shared_outer_loras",
                "base_model_fingerprint",
            ):
                if getattr(saved, name) != expected[name]:
                    raise NativeSourceError(
                        f"native checkpoint {name} does not match the loaded slot"
                    )
        source = (
            None
            if checkpoint is None
            else {
                "path": checkpoint,
                "load_optimizer": restore_optimizer,
                "parametrization_fields": self.fields(rank, alpha),
            }
        )
        result = await self.bridge.load_slot(slot, rank, alpha, source=source)
        if isinstance(result, dict) and result.get("error_category") == "user":
            raise NativeSourceError(result["error"])
        if not isinstance(result, dict) or "error" not in result:
            self.slots[slot] = (rank, alpha)
        return result

    async def capture(self, slot: int, path: str, *, include_optimizer: bool):
        rank, alpha = self.slots[slot]
        target = Path(path)
        method = self.bridge.save_slot if include_optimizer else self.bridge.export_slot
        # Spindle owns internal capture directories; tenant authorization remains in its outer metadata.
        written = await method(
            slot,
            root=str(target.parent),
            name=target.name,
            tenant_digest=None,
            parametrization_fields=self.fields(rank, alpha),
        )
        await asyncio.to_thread(
            self.finish_capture, target, written, include_optimizer=include_optimizer
        )

    def finish_capture(self, target: Path, written, *, include_optimizer: bool) -> None:
        expected = target.parent / f".{target.name}.{written.commit_id}"
        if written.path != expected or written.path.is_symlink():
            raise RuntimeError("native writer returned a directory outside its capture")
        try:
            refresh_native_capture(written.path)
            validate_native_files(written.path, require_state=include_optimizer)
            with closing(self.commit.lock_checkpoint_root(target.parent)):
                self.commit.commit_adapter_dir(
                    target.parent, target.name, written, overwrite=False
                )
                detach_native_capture(target, written.path)
        except BaseException:
            discard_native_capture(target, written.path)
            raise


def detach_native_capture(target: Path, physical: Path) -> None:
    """Transfer the committed directory to Spindle, whose captures must own their files."""
    if (
        physical.parent != target.parent
        or physical.is_symlink()
        or not physical.name.startswith(f".{target.name}.")
    ):
        raise RuntimeError("native capture has an unexpected physical directory")
    if not target.is_symlink() or target.resolve(strict=True) != physical.resolve(
        strict=True
    ):
        raise RuntimeError(
            "native capture link does not point to its committed directory"
        )
    target.unlink()
    try:
        physical.rename(target)
    except BaseException:
        target.symlink_to(physical.name, target_is_directory=True)
        raise


def discard_native_capture(target: Path, physical: Path) -> None:
    if target.is_symlink() and target.resolve() == physical.resolve():
        target.unlink()
    shutil.rmtree(physical, ignore_errors=True)
