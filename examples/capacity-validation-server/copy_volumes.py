"""Copy the shared server's weights and kernel cache into another Modal environment.

Volumes belong to one environment, so a new deployment would otherwise download
every base model again (about 6 TB) and recompile its kernels. This copies the
`spindle` environment's volumes file by file, one chunk per container input,
with as many CPU containers as Modal will run at once:

    uv run modal run -e capacity-validation examples/capacity-validation-server/copy_volumes.py

The destination environment is the one the app runs in. Files that already
exist there with the same size are skipped, so a rerun resumes an interrupted
copy. `--prefix Qwen/Qwen3.5-2B` copies one subtree of every volume.
"""

import os
import shutil
import time

import modal

SOURCE_ENVIRONMENT = "spindle"
# Volume name -> VolumeFS version, matching how Spindle creates each one.
# Checkpoints and the snapshot bulletin hold per-deployment state and are not copied.
VOLUMES = {"spindle-model-assets": None, "spindle-kernel-cache": 2}
# A chunk holds one large file, or many small files up to these limits.
CHUNK_BYTES = 1 << 30
CHUNK_FILES = 500

app = modal.App("spindle-copy-volumes")
image = modal.Image.debian_slim(python_version="3.12")


def source_volume(name):
    return modal.Volume.from_name(
        name, environment_name=SOURCE_ENVIRONMENT, version=VOLUMES[name]
    )


def destination_volume(name):
    return modal.Volume.from_name(name, create_if_missing=True, version=VOLUMES[name])


@app.function(
    image=image,
    cpu=2,
    memory=2048,
    timeout=4 * 60 * 60,
    retries=3,
    max_containers=1000,
    volumes={
        **{f"/src/{name}": source_volume(name) for name in VOLUMES},
        **{f"/dst/{name}": destination_volume(name) for name in VOLUMES},
    },
)
def copy_chunk(volume: str, paths: list[str]) -> int:
    copied = 0
    for path in paths:
        source = f"/src/{volume}/{path}"
        destination = f"/dst/{volume}/{path}"
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        # copy2 keeps mtimes, which Hugging Face's local-dir metadata checks
        # before trusting a downloaded file.
        shutil.copy2(source, destination, follow_symlinks=False)
        copied += os.path.getsize(destination)
    destination_volume(volume).commit()
    return copied


def files(volume, prefix):
    root = prefix.strip("/") or "/"
    for attempt in range(8):
        try:
            return {
                entry.path.lstrip("/"): entry.size
                for entry in volume.iterdir(root, recursive=True)
                if entry.type != modal.volume.FileEntryType.DIRECTORY
            }
        except modal.exception.NotFoundError:
            return {}
        except modal.exception.ResourceExhaustedError:
            # Listing is rate limited per workspace.
            time.sleep(2**attempt)
    raise RuntimeError(f"listing {root} stayed rate limited")


def chunks(sizes):
    chunk, chunk_bytes = [], 0
    for path, size in sorted(sizes.items(), key=lambda item: item[1], reverse=True):
        if chunk and (chunk_bytes + size > CHUNK_BYTES or len(chunk) >= CHUNK_FILES):
            yield chunk
            chunk, chunk_bytes = [], 0
        chunk.append(path)
        chunk_bytes += size
    if chunk:
        yield chunk


@app.local_entrypoint()
def main(prefix: str = ""):
    started = time.monotonic()
    for name in VOLUMES:
        source = files(source_volume(name), prefix)
        existing = files(destination_volume(name), prefix)
        missing = {
            path: size for path, size in source.items() if existing.get(path) != size
        }
        print(
            f"{name}: {len(source)} files, {sum(source.values()) / 1e9:.1f} GB; "
            f"copying {len(missing)} files, {sum(missing.values()) / 1e9:.1f} GB"
        )
        work = list(chunks(missing))
        # Function.map over no inputs does not return.
        total = (
            sum(copy_chunk.map([name] * len(work), work, order_outputs=False))
            if work
            else 0
        )
        print(f"{name}: copied {total / 1e9:.1f} GB in {len(work)} chunks")
        copied = files(destination_volume(name), prefix)
        wrong = [path for path, size in source.items() if copied.get(path) != size]
        if wrong:
            raise RuntimeError(f"{name}: {len(wrong)} files differ, e.g. {wrong[:5]}")
    print(f"done in {time.monotonic() - started:.0f}s")
