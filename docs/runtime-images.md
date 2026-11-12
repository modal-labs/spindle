# Runtime image ownership

Spindle keeps its Miles and SGLang base references in
`src/spindle/providers/modal/runtime_images.py`. Each entry records the original
upstream reference and the image that deployments use. Source overlays, model
patches, and the different SGLang versions remain in their existing image modules.

An upstream digest prevents a tag from changing the selected bytes. It does not
keep those bytes available if the upstream registry removes them. A retained copy
in a registry we control protects deployment rebuilds from that removal.

## Preserve the existing bases

Run **Preserve runtime images** in GitHub Actions. It copies every platform and
layer to `ghcr.io/<repository-owner>/spindle-runtimes/<image-name>`, using the
repository's `GITHUB_TOKEN` with `packages: write`. It needs no Docker Hub write
credentials. The token must be allowed to create and update those packages.

New GHCR packages are private by default. Make each package public in its package
settings, then rerun the workflow if its anonymous verification failed. Spindle's
Modal images pull these public bases without a registry secret. Keep the packages
public and exempt the preserved versions from deletion or cleanup while a
supported Spindle release references them.

The script resolves each current reference to a digest before copying. It copies
the full manifest list with `skopeo copy --all --preserve-digests`, then checks
that every destination has the same digest and can be read anonymously after
all copies finish. Failure produces no replacement lock file. Reruns read the current deployment references,
so an already-promoted owned copy does not require the upstream registry.

Download the `preserved-runtime-images` artifact and replace
`src/spindle/providers/modal/runtime_images.py` with its `runtime_images.py`.
Review and commit that file in a PR. Only the `image` fields should change; every
new reference must point to the owned registry and include `@sha256:...`.
The initial inventory retains upstream repositories until this publication and
verification has actually succeeded. The SGLang deployment references are pinned
to manifests verified on November 12, 2026. There is no automatic fallback to
upstream.

For a local run, install Skopeo, log in to the destination registry, and run:

```sh
python scripts/images/preserve.py \
  --namespace ghcr.io/modal-projects/spindle-runtimes \
  --output /tmp/preserved-runtime-images.py
```

SGLang tag-only references resolve to the bytes present at copy time. Compare
their resolved digest with the intended upstream release before promotion. A
preservation run cannot recover a dev image that has already been deleted.

## Build source candidates

Run **Build runtime image candidate** on a Linux x86-64 runner with the
`self-hosted` and `spindle-image-builder` labels, Docker Buildx, Python 3, and
sufficient disk for CUDA builds. The workflow has a six-hour timeout. Configure
that runner before dispatching the workflow.

For Miles, supply the full source commit in `modal-projects/miles`, a retained
SGLang base reference with a digest, and full SGLang and Megatron commits available
in their upstream repositories. The build uses Miles's Dockerfile from that
commit, replaces its SGLang `FROM` with the supplied base, clones the Miles
fork, and fetches the exact source commits. It never starts from a
`radixark/miles` image.

For SGLang, supply the full source commit in `modal-projects/sglang`, a
digest-pinned NVIDIA CUDA 13.0.3 cuDNN devel Ubuntu 24.04 base, and a digest-pinned
`astral-sh/uv` image. The build uses the checked-out source in local-source mode.
The preparation script rejects unsupported Dockerfile layouts rather than
silently ignoring a source pin.

The workflow publishes a uniquely tagged candidate and records its digest,
source inputs, and prepared Dockerfile as an artifact. These are new builds, not
byte-for-byte reproductions: upstream Dockerfiles can still resolve Python,
APT, Rust, and wheel dependencies at build time. Retain promoted candidate
images and the wheel artifacts needed to rebuild them. A source commit alone
does not freeze the complete CUDA stack.

Before promotion, make the candidate public and verify an anonymous pull. Check
imports and release commits, trainer forward/backward and optimizer steps,
adapter export and sampler loading, and a sampling request with the stock Tinker
SDK. Exercise the relevant GPU, model, LoRA overlap loading, and Rust routing
binding. Keep Miles v0.1.0, v0.1.1, and v0.1.2 separate, and keep DeepSeek's
SGLang v0.5.21 base separate from the v0.5.23 default. Record the tested commit,
hardware, and coverage limits in the promotion PR.

After changing a deployment pin, redeploy the affected server or worker apps.
Existing apps and snapshots keep their previous images. Retain those images
through rollback and while older releases remain supported. A rollback restores
the earlier digest in the inventory and redeploys the affected apps.
