"""SGLang worker entrypoint. The frontend never imports this GPU-only module."""

import json
import logging
import os
import sys

from sglang.launch_server import run_server
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree

from spindle.startup_timing import timeline
from spindle.telemetry.benchmark import enabled, start_collection


def main():
    timeline("sampler-sglang").mark("sglang_imports_done")
    load_plugins()
    settings = json.loads(sys.argv[2])
    if enabled():
        settings["enable_metrics"] = True
    args = ServerArgs(
        model_path=sys.argv[1],
        host="127.0.0.1",
        port=8001,
        **settings,
    )
    logging.basicConfig(level=args.log_level.upper())
    collection = start_collection(urls=("http://127.0.0.1:8001/metrics",))
    try:
        run_server(args)
    finally:
        collection.set()
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
