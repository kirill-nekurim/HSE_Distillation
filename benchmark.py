"""Замеры запросов к одной загруженной VLM: прогрев отдельно от повторов."""

import argparse
import csv
import gc
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    parser.add_argument("--image", type=Path, default=Path("test-small.jpg"))
    parser.add_argument("--output", type=Path, default=Path("results/local-4bit-small.csv"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Name one main object in the image. Answer with a single word.")
    args = parser.parse_args()
    if args.runs < 1 or args.max_tokens < 1:
        parser.error("Число повторов и лимит токенов должны быть положительными")
    if not args.image.is_file():
        parser.error(f"Изображение не найдено: {args.image}")

    import mlx.core as mx
    from PIL import Image
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    packages = {name: version(name) for name in ("mlx", "mlx-vlm", "transformers", "Pillow")}
    for name, value in packages.items():
        print(f"{name}: {value}", flush=True)
    with Image.open(args.image) as image:
        image_size = image.size
    print(f"Image: {args.image.name}, size: {image_size}", flush=True)

    print("Loading model...", flush=True)
    started = time.perf_counter()
    model, processor = load(args.model)
    config = load_config(args.model)
    mx.eval(model.parameters())
    mx.synchronize()
    load_seconds = time.perf_counter() - started
    print(f"Model loading: {load_seconds:.3f} s", flush=True)
    prompt = apply_chat_template(processor, config, args.prompt, num_images=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    fields = ["model", "image", "phase", "run", "load_seconds", "request_seconds", "mlx_peak_gib", "answer"]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for run in range(args.runs + 1):
            phase = "warmup" if run == 0 else "measured"
            print(f"Starting {phase} run {run}...", flush=True)
            gc.collect()
            mx.synchronize()
            mx.reset_peak_memory()
            started = time.perf_counter()
            result = generate(
                model, processor, prompt, [str(args.image.resolve())],
                max_tokens=args.max_tokens, temperature=0.0, verbose=False,
            )
            mx.synchronize()
            elapsed = time.perf_counter() - started
            text = getattr(result, "text", str(result)).strip()
            peak_gib = mx.get_peak_memory() / (1024 ** 3)
            row = dict(zip(fields, [args.model, args.image.name, phase, run, load_seconds, elapsed, peak_gib, text]))
            rows.append(row)
            writer.writerow(row)
            file.flush()
            print(f"{phase}: {elapsed:.3f} s | MLX peak: {peak_gib:.3f} GiB | answer: {text!r}", flush=True)

    times = [row["request_seconds"] for row in rows if row["phase"] == "measured"]
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "python": platform.python_version(), "packages": packages,
        "model": args.model, "image": args.image.name, "image_size": image_size,
        "prompt": args.prompt, "max_tokens": args.max_tokens, "temperature": 0.0,
        "measured_runs": args.runs, "warmup_runs": 1,
        "load_seconds": load_seconds, "median_seconds": statistics.median(times),
        "min_seconds": min(times), "max_seconds": max(times),
        "memory_note": "MLX active-allocation peak, not total process or system RAM",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nMedian request time: {statistics.median(times):.3f} s")
    print(f"Min/max: {min(times):.3f} / {max(times):.3f} s")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
