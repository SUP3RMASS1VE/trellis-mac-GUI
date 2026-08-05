#!/usr/bin/env bash
#
# Set up TRELLIS.2 for Apple Silicon.
# Creates a venv, installs dependencies, clones the repo, and applies patches.
#

set -euo pipefail
cd "$(dirname "$0")"

echo "=== TRELLIS.2 for Apple Silicon — Setup ==="
echo

# ---------------------------------------------------------------------------
# Vendored sources. TRELLIS.2 and everything under deps/ ship inside this
# repository, so setup does no cloning — the only network I/O is pip installs
# and the model weights fetched on first run.
#
# Upstream origins, for reference when updating a vendored tree:
#   TRELLIS.2            github.com/microsoft/TRELLIS.2
#   deps/utils3d         github.com/EasternJournalist/utils3d @ 9a4eb15
#   deps/mtlbvh          github.com/pedronaugusto/mtlbvh
#   deps/mtldiffrast     github.com/pedronaugusto/mtldiffrast
#   deps/mtlmesh         github.com/pedronaugusto/mtlmesh      (module: cumesh)
#   deps/mtlgemm         github.com/pedronaugusto/mtlgemm      (module: flex_gemm)
#   deps/trellis2-apple  github.com/pedronaugusto/trellis2-apple (o_voxel fork)
# ---------------------------------------------------------------------------
DEPS_DIR="deps"

missing=0
for tree in \
    "TRELLIS.2" \
    "$DEPS_DIR/utils3d" \
    "$DEPS_DIR/mtlbvh" \
    "$DEPS_DIR/mtldiffrast" \
    "$DEPS_DIR/mtlmesh" \
    "$DEPS_DIR/mtlgemm" \
    "$DEPS_DIR/trellis2-apple"
do
    if [ ! -d "$tree" ]; then
        echo "  MISSING: $tree"
        missing=1
    fi
done

if [ "$missing" = "1" ]; then
    echo
    echo "Vendored sources are incomplete. These directories ship with the repo —"
    echo "if they are absent the checkout is partial (e.g. a sparse clone, or a"
    echo "zip export that dropped them). Re-clone the repository and retry."
    exit 1
fi

echo "Vendored sources: OK (TRELLIS.2 + $(ls -1 "$DEPS_DIR" | wc -l | tr -d ' ') deps)"
echo

# Check Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Warning: This project requires Apple Silicon (M1 or later)."
fi

# Create venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    if command -v uv &>/dev/null; then
        uv venv .venv --python python3.11
    else
        python3 -m venv .venv
    fi
fi

source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
DEPS="torch torchvision torchaudio transformers accelerate huggingface_hub safetensors pillow numpy trimesh scipy tqdm easydict kornia timm imageio opencv-python-headless xatlas fast-simplification gradio einops"
if command -v uv &>/dev/null; then
    PIP="uv pip install"
else
    PIP="pip install"
fi
$PIP $DEPS
$PIP "$DEPS_DIR/utils3d"

# ---------------------------------------------------------------------------
# Xcode Metal Toolchain. The mtl* packages below compile .metal sources, which
# needs this component — it is NOT part of a stock Xcode install. Without it,
# every Metal build fails and we fall back to the pure-Python KDTree baker
# (minutes per bake instead of seconds).
#
# Skip with SKIP_METAL=1 (whole Metal stack) or SKIP_METAL_TOOLCHAIN=1
# (assume the toolchain is handled externally).
# ---------------------------------------------------------------------------
metal_toolchain_installed() {
    xcodebuild -showComponent MetalToolchain 2>/dev/null | grep -qi "Status:[[:space:]]*installed"
}

ensure_metal_toolchain() {
    if metal_toolchain_installed; then
        echo "Metal Toolchain: already installed"
        return 0
    fi

    # -downloadComponent needs a full Xcode; Command Line Tools alone can't do it.
    local dev_dir
    dev_dir="$(xcode-select -p 2>/dev/null || true)"
    if [[ "$dev_dir" != *Xcode.app* ]]; then
        echo "Metal Toolchain: requires full Xcode, but xcode-select points at:"
        echo "    ${dev_dir:-<nothing>}"
        echo "  Install Xcode from the App Store, then:"
        echo "    sudo xcode-select -s /Applications/Xcode.app"
        return 1
    fi

    echo "Metal Toolchain: not installed — downloading (several GB, one time)..."
    if xcodebuild -downloadComponent MetalToolchain; then
        echo "Metal Toolchain: installed"
        return 0
    fi

    echo "Metal Toolchain: download failed. It may need elevated privileges:"
    echo "    sudo xcodebuild -downloadComponent MetalToolchain"
    echo "  Continuing without it — the KDTree texture baker will be used."
    return 1
}

# Optional Metal acceleration for texture baking.
# Without these, we fall back to a pure-Python KDTree-based texture baker.
#
# --no-build-isolation is critical: these packages need torch at build time,
# and uv's default isolated build env has no torch installed.
if [ "${SKIP_METAL:-0}" != "1" ]; then
    if [ "${SKIP_METAL_TOOLCHAIN:-0}" != "1" ]; then
        ensure_metal_toolchain || true
    fi
    # PyTorch's MPS headers require macOS 12.0+. Some Python builds (e.g. uv's
    # prebuilt binaries) set -mmacosx-version-min=11.0 which makes the compiler
    # reject the MPS headers with -Werror. Override to 12.0 for the Metal builds.
    export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-12.0}
    echo
    echo "Installing Metal backends for texture baking (set SKIP_METAL=1 to skip)..."
    PIP_NB="$PIP --no-build-isolation"
    # Build deps required by the Metal packages' setup.py
    $PIP setuptools wheel pybind11
    $PIP_NB "$DEPS_DIR/mtlbvh"      || echo "  mtlbvh install failed — continuing without Metal BVH"
    $PIP_NB "$DEPS_DIR/mtldiffrast" || echo "  mtldiffrast install failed — continuing without Metal rasterizer"
    $PIP_NB "$DEPS_DIR/mtlmesh"     || echo "  mtlmesh install failed — continuing without Metal mesh ops"
    # mtlgemm provides flex_gemm.ops.grid_sample. The Metal baker in
    # o_voxel.postprocess prefers this over a torch.nn.functional.grid_sample
    # fallback, and the flex_gemm sparse sampling produces noticeably cleaner
    # texture baking (no concentric ring artifacts on curved surfaces).
    $PIP_NB "$DEPS_DIR/mtlgemm"     || echo "  mtlgemm install failed — baker will use a lower-quality torch.nn.functional.grid_sample fallback"
    # Pedro Naugusto's o_voxel CPU fork — exposes o_voxel.postprocess.to_glb
    # which wraps the Metal stack. Install last so its deps already present.
    $PIP_NB "$DEPS_DIR/trellis2-apple/o-voxel" \
        || echo "  o_voxel (Apple fork) install failed — falling back to KDTree baker"
fi

# Apply source patches (this also installs stubs and backends)
echo "Applying MPS compatibility patches..."
python3 patches/mps_compat.py

# Check HuggingFace auth
echo
if python3 -c "from huggingface_hub import get_token; assert get_token()" 2>/dev/null; then
    echo "HuggingFace auth: OK"
else
    echo "HuggingFace auth: not logged in (fine — all required weights are ungated)."
    echo "Log in for higher download rate limits:"
    echo "  hf auth login"
fi

echo
echo "=== Setup complete ==="
echo "Activate the environment:  source .venv/bin/activate"
echo "Generate a 3D model:       python generate.py path/to/image.png"
