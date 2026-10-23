"""Export the experimental C1/C4 DFlash2 catalogs with 32K, 64K or 128K capacity.

Run from the repository root with its preparation environment. This is a build
step; Python is not involved in native request execution.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

TARGET = "nvidia/Qwen3.8-27B-NVFP4"
TARGET_REVISION = "482ca0f3832238542f8f5295dde86b5f22711d80"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, choices=(32768, 65536, 131072), default=65536)
    parser.add_argument("--concurrency", type=int, choices=(1, 4), default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--target-snapshot", type=Path, help="local copy of the pinned target")
    parser.add_argument("--draft-snapshot", type=Path, help="local copy of the pinned NVFP4 drafter")
    parser.add_argument("--compile-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new package path")
    if args.compile_workers < 1:
        parser.error("compile-workers must be positive")
    return args


def main(argv=None):
    args = arguments(argv)
    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply
    from ignition.cli.preparation import Host

    configuration = Configuration.load(args.config)
    devices = Host.discover().gpus
    if not 0 <= args.gpu < len(devices):
        raise ValueError("GPU index is outside the visible device list")
    gpu = devices[args.gpu]
    if gpu.capability != (12, 0):
        raise ValueError("this experimental export example is qualified on SM120")
    apply(configuration, (gpu,))

    import torch
    from huggingface_hub import snapshot_download
    from ignition.cli.serve import _publisher
    from ignition.preparation.model_package import export_package
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_storage import PackageStorage
    from ignition.recipes import resolve
    from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig, prepare
    from ignition.recipes.qwen38_dense.dflash2_config import NVFP4_REPOSITORY, NVFP4_REVISION

    cache = configuration.directory("hub_cache")
    snapshot = args.target_snapshot or Path(snapshot_download(TARGET, revision=TARGET_REVISION, cache_dir=cache))
    draft = args.draft_snapshot or Path(snapshot_download(NVFP4_REPOSITORY,
        revision=NVFP4_REVISION, cache_dir=cache, allow_patterns=("config.json", "model.safetensors")))
    # Local overrides must be copies of these exact checkpoints; no conversion is performed.
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    images = ModuleImages()
    model = None
    storage = None
    progress = lambda message: print(message, flush=True)
    try:
        with images:
            storage = PackageStorage()
            with storage:
                model = prepare(snapshot, PreparationConfig(mtp=True,
                    speculative_proposal_source="dflash2", draft_checkpoint=draft,
                    speculative_widths=(1, 4, 8), context=args.context, prefill_chunk_size=1024,
                    max_active=args.concurrency, speculative_token_capacity=args.concurrency * 8,
                    snapshot_budget_bytes=1 << 30, headroom_bytes=2 << 30,
                    compile_workers=args.compile_workers, compile_decoder=True, retain_graphs=True),
                    progress=progress, package_storage=storage)
        resolved = resolve("qwen3.8-27b-dflash2-nvfp4", cache)
        publisher = _publisher(configuration, SimpleNamespace(official_source=None), resolved, cache)
        options = dict(model_id=TARGET, served_names=["Qwen3.8-27B"],
            reserve_bytes=2 << 30, queue_capacity=4 * args.concurrency,
            frontend_workers=args.concurrency, max_body_bytes=args.context * 32,
            context_hint=f"{args.context // 1024}K experimental DFlash2",
            capacity_hint=f"C{args.concurrency} experimental DFlash2")
        export_package(model, images, storage, args.output, options=options,
            publisher=publisher, progress=progress)
        print(f"Exported {args.output}; preserve its sibling payload files.", flush=True)
    finally:
        if model is not None:
            model.close()
        if storage is not None:
            for region in storage.host_regions:
                region.close()


if __name__ == "__main__":
    main()
