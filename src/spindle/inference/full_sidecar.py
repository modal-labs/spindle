from __future__ import annotations

import argparse
import time

import uvicorn
from pathlib import Path

from stitch.engines.sglang import SGLangEngine
from stitch.service import create_app
from stitch.sync import Reconciler

from spindle.telemetry.benchmark import record as record_benchmark_metrics
from stitch.types import VersionRef

from .fft_bulletin import (
    FFTSnapshotBulletin,
    FFTSnapshotStore,
    PinnedFFTSnapshotStore,
)


class MeasuredReconciler(Reconciler):
    async def _reconcile_once(self):
        previous = self.applied
        result = await super()._reconcile_once()
        if self.applied is not None and self.applied != previous:
            record_benchmark_metrics(
                {
                    "weight_sync.ready_unix_s": time.time(),
                    "weight_sync.reconcile": self.metrics,
                },
                {
                    "spindle.model_id": str(self.applied.run_id),
                    "publish_version": self.applied.version,
                },
            )
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--bulletin-root", required=True)
    parser.add_argument("--bulletin-volume", default="")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pinned-version", type=int)
    parser.add_argument("--scoped-registry")
    args = parser.parse_args()

    refresh = None
    if args.bulletin_volume:
        import modal

        volume = modal.Volume.from_name(args.bulletin_volume, version=2)
        refresh = volume.reload
    bulletin = FFTSnapshotBulletin(
        Path(args.bulletin_root),
        refresh=refresh,
    )
    store = FFTSnapshotStore(bulletin, args.run_id)
    if args.pinned_version is not None:
        store = PinnedFFTSnapshotStore(
            bulletin,
            VersionRef(args.run_id, args.pinned_version),
        )
    engine = SGLangEngine(
        args.upstream_url,
        delta_update_mode="cpu",
    )
    if args.scoped_registry:
        import modal
        from .scoped_sidecar import AssignedSnapshotStore, serve_assigned

        registry = modal.Dict.from_name(args.scoped_registry)
        store = AssignedSnapshotStore(bulletin, args.run_id, registry)
        serve_assigned(
            store,
            engine,
            run_id=args.run_id,
            registry=registry,
            host=args.host,
            port=args.port,
        )
        return
    reconciler = MeasuredReconciler(store=store, engine=engine, run_id=args.run_id)
    uvicorn.run(
        create_app(reconciler.gate, reconciler, engine),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
