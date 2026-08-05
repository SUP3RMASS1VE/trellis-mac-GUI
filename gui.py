"""
Local Gradio GUI for TRELLIS.2 image-to-3D on Apple Silicon.

Runs the same MPS code path as generate.py, but keeps the pipeline warm between
runs so you only pay the ~100s load once per process.

    python gui.py            # http://127.0.0.1:7860

The server binds to localhost only and has no authentication. Do not expose it
to a network (--host 0.0.0.0 / --share) unless you add access control.
"""

# runner sets up MPS/backend env vars and sys.path; it must be imported first.
import runner

import argparse
import os
import queue
import random
import re
import sys
import threading
import time
from datetime import datetime

import gradio as gr

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "outputs")
EXAMPLE_DIR = os.path.join(ROOT, "TRELLIS.2", "assets", "example_image")
MAX_SEED = 2**31 - 1

# tqdm bars arrive on stderr as \r-terminated redraws; recognise them so they
# update a single status line instead of flooding the log.
_TQDM_RE = re.compile(r"\d+%\|")


class _StreamTap:
    """
    Forwards a stream to the real one while feeding it into the UI.

    Progress-bar redraws go to a single mutable status slot; everything else is
    appended to the log queue. Only one generation runs at a time
    (concurrency_limit=1), so a process-wide swap is safe here.

    Both stdout and stderr are tapped: samplers write tqdm bars to stderr, while
    the texture baker reports its stages with plain print() to stdout.
    """

    def __init__(self, real, log_q, status):
        self._real = real
        self._log_q = log_q
        self._status = status
        self._buf = ""

    def write(self, text):
        self._real.write(text)
        self._buf += text
        while True:
            match = re.search(r"[\r\n]", self._buf)
            if not match:
                break
            line, self._buf = self._buf[: match.start()], self._buf[match.end():]
            line = line.rstrip()
            if not line:
                continue
            if _TQDM_RE.search(line):
                self._status["line"] = line
                self._status["at"] = time.time()
            else:
                self._log_q.put(line)
        return len(text)

    def flush(self):
        self._real.flush()

    def isatty(self):
        return False


def resolve_seed(randomize, seed):
    return random.randint(0, MAX_SEED) if randomize else int(seed)


# Weights fetched on first run. Sizes are approximate on-disk footprints.
MODEL_REPOS = [
    ("microsoft/TRELLIS.2-4B", "pipeline", "14 GB"),
    ("camenduru/dinov3-vitl16-pretrain-lvd1689m", "image conditioning", "1.1 GB"),
    ("ZhengPeng7/BiRefNet", "background removal", "424 MB"),
]


def _repo_cached(repo_id):
    """True if this repo already has a HuggingFace cache directory."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return False
    return os.path.isdir(os.path.join(HF_HUB_CACHE, "models--" + repo_id.replace("/", "--")))


def uncached_repos():
    return [(r, what, size) for r, what, size in MODEL_REPOS if not _repo_cached(r)]


def pipeline_status_text():
    """
    Describe pipeline state for the Status box.

    The pipeline lives in this Python process, so every app restart needs a
    fresh load — worth saying out loud, since the load is ~75s.
    """
    if runner.is_pipeline_loaded():
        return "Pipeline loaded on MPS — ready to generate."

    missing = uncached_repos()
    if missing:
        total = ", ".join(f"{what} {size}" for _, what, size in missing)
        return (f"Pipeline not loaded. First run downloads ~15 GB of weights ({total}) "
                f"to ~/.cache/huggingface — this can take a while on a slow connection.")
    return ("Pipeline not loaded. Weights are cached, so loading takes ~75s. "
            "It runs automatically on your first generation.")


def load_pipeline_ui():
    """
    Load the pipeline up front so the first generation isn't ~75s slower.

    Downloads any missing weights on the way, which on a first ever run is the
    full ~15 GB.
    """
    if runner.is_pipeline_loaded():
        return "Pipeline already loaded on MPS — ready to generate."

    downloading = bool(uncached_repos())
    t0 = time.time()
    runner.load_pipeline()
    elapsed = time.time() - t0
    note = " (weights downloaded and cached for next time)" if downloading else ""
    return f"Pipeline loaded on MPS in {elapsed:.0f}s{note}. Reloads on each app restart."


def loading_placeholder():
    if runner.is_pipeline_loaded():
        return "Pipeline already loaded on MPS — ready to generate."
    if uncached_repos():
        return "Downloading weights (~15 GB on first run) then loading onto MPS — watch the terminal for progress..."
    return "Loading pipeline onto MPS (~75s)..."


def image_to_3d(
    image,
    seed,
    randomize_seed,
    pipeline_type,
    texture_size,
    bake_texture,
    steps,
):
    """
    Generator: streams log output while generation runs on a worker thread.

    Yields (model3d, files, log, status) tuples.
    """
    if image is None:
        raise gr.Error("Upload an image first.")

    seed = resolve_seed(randomize_seed, seed)
    steps = int(steps) if steps else None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = os.path.join(OUTPUT_DIR, f"trellis_{stamp}")

    log_q = queue.Queue()
    # "line" is the newest activity shown in the Status box; "at" is when it
    # arrived, used to detect a stage that has gone quiet.
    status = {"line": "", "at": time.time()}
    box = {}

    def worker():
        real_stdout, real_stderr = sys.stdout, sys.stderr
        sys.stdout = _StreamTap(real_stdout, log_q, status)
        sys.stderr = _StreamTap(real_stderr, log_q, status)
        try:
            box["result"] = runner.generate(
                image,
                seed=seed,
                pipeline_type=pipeline_type,
                texture_size=int(texture_size),
                texture=bake_texture,
                steps=steps,
                output=output,
                log=log_q.put,
            )
        except BaseException as e:  # surfaced to the UI below
            box["error"] = e
        finally:
            sys.stdout, sys.stderr = real_stdout, real_stderr
            log_q.put(None)

    lines = [f"seed={seed}  pipeline={pipeline_type}  texture={'off' if not bake_texture else texture_size}"]
    if not runner.is_pipeline_loaded():
        lines.append("Cold start: pipeline load adds ~100s (once per process).")

    thread = threading.Thread(target=worker, daemon=True)
    t0 = time.time()
    thread.start()

    def render():
        elapsed = time.time() - t0
        head = f"Running — {elapsed:.0f}s elapsed"
        tail = status["line"]
        if not tail:
            return "\n".join(lines), head
        # Some stages (Metal BVH build, xatlas packing) print nothing for a
        # while. Say so, rather than leaving a finished progress bar on screen
        # looking stuck.
        quiet = time.time() - status["at"]
        if quiet > 5:
            tail += f"\n(still working — no new output for {quiet:.0f}s)"
        return "\n".join(lines), f"{head}\n{tail}"

    log_text, status_text = render()
    yield None, None, log_text, status_text

    done = False
    while not done:
        try:
            item = log_q.get(timeout=0.5)
            if item is None:
                done = True
            else:
                lines.append(item)
                # Surface the newest phase message in Status too, so a finished
                # progress bar never lingers there as the apparent current state.
                status["line"] = item
                status["at"] = time.time()
        except queue.Empty:
            pass
        log_text, status_text = render()
        yield gr.skip(), gr.skip(), log_text, status_text

    thread.join()
    elapsed = time.time() - t0

    if "error" in box:
        err = box["error"]
        lines.append(f"FAILED: {err}")
        yield None, None, "\n".join(lines), f"Failed after {elapsed:.0f}s"
        if isinstance(err, runner.WatchdogError):
            raise gr.Error(str(err), duration=30)
        raise gr.Error(f"{type(err).__name__}: {err}", duration=30)

    result = box["result"]
    files = [result["glb"], result["obj"]]
    if result.get("basecolor"):
        files.append(result["basecolor"])

    summary = (
        f"Done in {elapsed:.0f}s — {result['vertices']:,} vertices, "
        f"{result['faces']:,} triangles  (gen {result['gen_time']:.0f}s, "
        f"bake {result['bake_time']:.0f}s, seed {seed})"
    )
    lines.append(summary)
    yield result["glb"], files, "\n".join(lines), summary


def example_images(limit=24):
    if not os.path.isdir(EXAMPLE_DIR):
        return []
    names = sorted(f for f in os.listdir(EXAMPLE_DIR) if f.lower().endswith((".webp", ".png", ".jpg", ".jpeg")))
    return [[os.path.join(EXAMPLE_DIR, n)] for n in names[:limit]]


def build_ui():
    with gr.Blocks(title="TRELLIS.2 on Apple Silicon") as demo:
        gr.Markdown(
            "## TRELLIS.2 image-to-3D on Apple Silicon\n"
            "Upload an image and generate a textured GLB. Background removal runs "
            "automatically. Keep the tab open — progress streams below.\n\n"
            "**First run downloads ~15 GB of model weights** to `~/.cache/huggingface` "
            "(pipeline, image conditioning, background removal). That happens once. "
            "After that the pipeline loads from disk in ~75s, and because it lives in "
            "this Python process it **loads again every time you restart the app** — "
            "either on your first generation, or up front with *Load pipeline*."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=340):
                image = gr.Image(label="Input image", type="pil", image_mode="RGBA",
                                 sources=["upload", "clipboard"], height=340)
                pipeline_type = gr.Radio(runner.PIPELINE_TYPES, value="512",
                                         label="Pipeline resolution",
                                         info="512 is fastest; 1024 variants are slower and use more memory")
                bake_texture = gr.Checkbox(True, label="Bake PBR textures",
                                           info="Off exports geometry only (much faster)")
                texture_size = gr.Radio([str(s) for s in runner.TEXTURE_SIZES], value="1024",
                                        label="Texture size")
                with gr.Row():
                    seed = gr.Number(42, label="Seed", precision=0, minimum=0, maximum=MAX_SEED)
                    randomize_seed = gr.Checkbox(False, label="Randomize")
                steps = gr.Slider(0, 30, value=0, step=1, label="Sampler steps override",
                                  info="0 uses the pipeline default (12). Lower is faster, rougher.")
                generate_btn = gr.Button("Generate 3D model", variant="primary")
                load_btn = gr.Button("Load pipeline", variant="secondary")
                gr.Markdown(
                    "<sub>Optional. Downloads weights on first ever run (~15 GB), then "
                    "loads in ~75s. Needed once per app start — otherwise the first "
                    "generation pays for it.</sub>"
                )

            with gr.Column(scale=2):
                status = gr.Textbox(label="Status", value="Checking model cache...",
                                    lines=3, interactive=False)
                model = gr.Model3D(label="Generated GLB", height=460, display_mode="solid",
                                   clear_color=(0.25, 0.25, 0.25, 1.0))
                files = gr.Files(label="Downloads (GLB / OBJ)")
                log = gr.Textbox(label="Log", lines=14, max_lines=14, interactive=False,
                                 autoscroll=True)

        examples = example_images()
        if examples:
            gr.Examples(examples=examples, inputs=[image], examples_per_page=12,
                        label="Example images (from TRELLIS.2 assets)")

        bake_texture.change(lambda on: gr.update(interactive=on),
                            inputs=[bake_texture], outputs=[texture_size])

        generate_btn.click(
            image_to_3d,
            inputs=[image, seed, randomize_seed, pipeline_type, texture_size, bake_texture, steps],
            outputs=[model, files, log, status],
            concurrency_limit=1,
        )
        # Report cache/load state as soon as the page opens, so the ~15 GB first
        # download and the per-restart reload are never a surprise.
        demo.load(pipeline_status_text, outputs=[status])

        load_btn.click(
            loading_placeholder, outputs=[status]
        ).then(
            load_pipeline_ui, outputs=[status], concurrency_limit=1
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio GUI for TRELLIS.2 on Apple Silicon")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--preload", action="store_true",
                        help="Load the pipeline at startup instead of on first generation")
    parser.add_argument("--open", action="store_true", help="Open the browser on launch")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.preload:
        runner.load_pipeline()

    demo = build_ui()
    demo.queue(max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=args.open,
        theme=gr.themes.Soft(),
        show_error=True,
    )


if __name__ == "__main__":
    main()
