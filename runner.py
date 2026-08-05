"""
Shared TRELLIS.2-on-Apple-Silicon runtime.

Holds the environment bootstrap, a process-wide pipeline cache, and the
generate/bake logic used by both the CLI (`generate.py`) and the Gradio GUI
(`gui.py`).

IMPORTANT: importing this module sets MPS/backend environment variables and
must happen before torch is imported anywhere else in the process.
"""

import os
import sys

# Set up backends before any TRELLIS imports. Use setdefault so the caller
# can override from the environment. Default conv backend is flex_gemm since
# Pedro Naugusto's mtlgemm fix (zero-copy on MPS, fp16/bf16 native); fall
# back to conv_none if flex_gemm isn't importable for some reason.
# MPS fallback MUST be set before torch is imported anywhere (including
# transitively via flex_gemm). Without this, segment_reduce and a few other
# ops crash instead of falling back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
try:
    import flex_gemm  # noqa: F401
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
except (ImportError, RuntimeError):
    # ImportError: package not installed (SKIP_METAL=1 or install failed).
    # RuntimeError: metallib load failure — flex_gemm ships an MSL 4.0
    # metallib which only loads on macOS 26+. On older macOS the import
    # itself raises "Failed to load metallib ... language version 4.0
    # which is not supported on this OS". Fall back to conv_none either way.
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add paths. stubs/ is appended (not prepended) so a pip-installed o_voxel
# wins over our package stub — the flat override module o_voxel_override_convert
# is still discoverable either way because it doesn't collide with any package.
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRELLIS.2"))
sys.path.append(os.path.join(_ROOT, "stubs"))

import time

import torch
from PIL import Image as PILImage

PIPELINE_TYPES = ["512", "1024", "1024_cascade"]
TEXTURE_SIZES = [512, 1024, 2048]

# Target face count before texture baking. The Metal BVH builder is unstable
# on 800K+ face inputs, and xatlas is very slow on them.
BAKE_TARGET_FACES = 200000

WATCHDOG_HELP = (
    "The decoder produced an empty mesh.\n"
    "On Apple Silicon this is almost always the macOS GPU watchdog\n"
    "killing a long-running Metal kernel in the SLat decoder. The Metal\n"
    "error prints to stderr (look for\n"
    "'kIOGPUCommandBufferCallbackErrorImpactingInteractivity') but does\n"
    "not raise a Python exception, so execution continues with empty\n"
    "tensors and crashes downstream.\n"
    "\n"
    "Workarounds, cheapest first:\n"
    "  1. Run headless — close the lid / unplug external displays and\n"
    "     re-run over SSH. The watchdog tightens with WindowServer load.\n"
    "  2. MTL_CAPTURE_ENABLED=1 python generate.py ...   (extends the\n"
    "     watchdog timeout as a side effect of Metal-debugger mode)\n"
    "  3. SPARSE_CONV_BACKEND=none python generate.py ... (slower path,\n"
    "     may not help if a single dispatch is the offender)\n"
    "\n"
    "Tracking issue: https://github.com/shivampkumar/trellis-mac/issues"
)

# Signatures of watchdog-induced corruption:
#   IndexError: max(): Expected reduction dim 0 to have non-zero size
#     — empty SparseTensor in decode_latent's spatial_shape calc
#   AssertionError: BVH needs at least 8 triangles, got 0
#     — empty mesh propagating into o_voxel.postprocess.to_glb
_WATCHDOG_SIGNATURES = ("non-zero size", "BVH needs at least 8 triangles")


class WatchdogError(RuntimeError):
    """Raised when generation output was corrupted by the macOS GPU watchdog."""

    def __init__(self, message=WATCHDOG_HELP):
        super().__init__(message)


_PIPELINE = None


def load_pipeline(log=print):
    """Load (and cache for the process) the TRELLIS.2 pipeline on MPS."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    log("Loading pipeline (first load takes ~100s)...")
    t0 = time.time()
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    log(f"Loaded in {time.time() - t0:.0f}s")

    pipeline.to(torch.device("mps"))
    log("Device: MPS")

    _PIPELINE = pipeline
    return _PIPELINE


def is_pipeline_loaded():
    return _PIPELINE is not None


def generate(
    image,
    seed=42,
    pipeline_type="512",
    texture_size=1024,
    texture=True,
    steps=None,
    output="output_3d",
    log=print,
):
    """
    Generate a 3D mesh from a single image.

    Args:
        image: PIL image or path to an image file.
        seed: random seed.
        pipeline_type: one of PIPELINE_TYPES.
        texture_size: PBR texture resolution, one of TEXTURE_SIZES.
        texture: bake PBR textures; when False export geometry only.
        steps: override sampler steps for all three flow phases.
        output: output path without extension.
        log: callable taking one string, used for progress reporting.

    Returns:
        dict with keys: glb, obj, basecolor (optional), vertices, faces,
        gen_time, bake_time.

    Raises:
        WatchdogError: generation output was empty (see WATCHDOG_HELP).
    """
    pipeline = load_pipeline(log=log)

    if not isinstance(image, PILImage.Image):
        image = PILImage.open(image)
    log(f"Input image: {image.size[0]}x{image.size[1]}")

    log(f"Generating 3D model (pipeline={pipeline_type}, seed={seed})...")
    t0 = time.time()
    sampler_overrides = {"steps": steps} if steps else {}

    try:
        outputs = pipeline.run(
            image,
            seed=seed,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params=sampler_overrides,
            shape_slat_sampler_params=sampler_overrides,
            tex_slat_sampler_params=sampler_overrides,
        )
    except (IndexError, AssertionError) as e:
        if any(sig in str(e) for sig in _WATCHDOG_SIGNATURES):
            raise WatchdogError() from e
        raise

    gen_time = time.time() - t0

    mesh_out = outputs[0] if isinstance(outputs, list) else outputs
    verts = mesh_out.vertices.cpu().numpy()
    faces = mesh_out.faces.cpu().numpy()
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise WatchdogError()

    log(f"Mesh: {verts.shape[0]:,} vertices, {faces.shape[0]:,} triangles")
    log(f"Generation time: {gen_time:.1f}s")

    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)

    glb_path = f"{output}.glb"
    has_voxels = getattr(mesh_out, "attrs", None) is not None
    result = {"vertices": int(verts.shape[0]), "faces": int(faces.shape[0]),
              "gen_time": gen_time, "bake_time": 0.0, "glb": glb_path}

    if has_voxels and texture:
        t_bake = time.time()
        basecolor = _bake(mesh_out, verts, faces, glb_path, int(texture_size), output, log)
        result["bake_time"] = time.time() - t_bake
        if basecolor:
            result["basecolor"] = basecolor
        log(f"Bake time: {result['bake_time']:.0f}s")
    else:
        # Fallback: geometry only.
        log("Exporting geometry only (no PBR textures)...")
        import trimesh

        trimesh.Trimesh(vertices=verts, faces=faces).export(glb_path)
        log(f"Saved: {glb_path}")

    obj_path = f"{output}.obj"
    with open(obj_path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    log(f"Saved: {obj_path}")
    result["obj"] = obj_path

    return result


def _patch_grid_sample_fallback(o_voxel_postprocess):
    """
    o_voxel's _grid_sample_3d fallback returns [B*C, M] but the bake consumes
    it as [M, C]. Patch it to transpose before the reshape. We avoid installing
    flex_gemm itself because its import slows the diffusion hot path ~10x on MPS.
    """
    import torch.nn.functional as F

    def _gs3d_fix(feats, coords, shape, grid, mode="trilinear"):
        B, C = shape[0], shape[1]
        D, H, W = shape[2], shape[3], shape[4]
        device = feats.device
        dense_vol = torch.zeros(B, C, D, H, W, dtype=feats.dtype, device=device)
        batch_idx = coords[:, 0].long()
        cx = coords[:, 1].long(); cy = coords[:, 2].long(); cz = coords[:, 3].long()
        dense_vol[batch_idx, :, cx, cy, cz] = feats
        grid_norm = torch.stack([
            grid[..., 2] / (W - 1) * 2 - 1,
            grid[..., 1] / (H - 1) * 2 - 1,
            grid[..., 0] / (D - 1) * 2 - 1,
        ], dim=-1).reshape(B, 1, 1, -1, 3)
        sampled = F.grid_sample(
            dense_vol, grid_norm, mode="bilinear",
            align_corners=True, padding_mode="border",
        )
        M = grid.shape[1]
        return sampled.reshape(B, C, M).permute(0, 2, 1).reshape(B * M, C)

    o_voxel_postprocess._grid_sample_3d = _gs3d_fix


def _bake(mesh_out, verts, faces, glb_path, tex_size, output, log):
    """Bake PBR textures and export a GLB. Returns a base-color PNG path or None."""
    # Try Metal-accelerated bake via o_voxel + mtldiffrast if available.
    # Catch AttributeError too: our stubs/o_voxel/ stub has no .postprocess
    # submodule, so a shadowing stub package trips getattr, not import.
    try:
        import o_voxel.postprocess

        use_metal = (
            getattr(o_voxel.postprocess, "_BACKEND", None) == "metal"
            and getattr(o_voxel.postprocess, "_HAS_DR", False)
        )
        if use_metal and not getattr(o_voxel.postprocess, "_HAS_FLEX_GEMM", False):
            _patch_grid_sample_fallback(o_voxel.postprocess)
    except (ImportError, AttributeError):
        use_metal = False

    if use_metal:
        try:
            log(f"Baking PBR textures via Metal ({tex_size}x{tex_size})...")
            import o_voxel

            # Pre-simplify mesh to avoid mtlbvh crash on large meshes.
            import fast_simplification

            target_faces = min(BAKE_TARGET_FACES, len(faces))
            if len(faces) > target_faces:
                ratio = 1.0 - (target_faces / len(faces))
                log(f"  Simplifying mesh: {len(faces):,} -> ~{target_faces:,} faces")
                simp_verts, simp_faces = fast_simplification.simplify(verts, faces, ratio)
                simp_verts_t = torch.from_numpy(simp_verts).float()
                simp_faces_t = torch.from_numpy(simp_faces.astype("int32"))
            else:
                simp_verts_t = mesh_out.vertices
                simp_faces_t = mesh_out.faces

            # Move all mesh tensors to CPU — o_voxel.to_glb mixes a device-neutral
            # AABB tensor with mesh tensors; keep everything on CPU to avoid mismatch.
            glb = o_voxel.postprocess.to_glb(
                vertices=simp_verts_t.cpu(),
                faces=simp_faces_t.cpu(),
                attr_volume=mesh_out.attrs.cpu(),
                coords=mesh_out.coords.cpu(),
                attr_layout=mesh_out.layout,
                voxel_size=mesh_out.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=target_faces,
                texture_size=tex_size,
                verbose=True,
            )
            glb.export(glb_path)
            log(f"  Saved: {glb_path}")
            return None
        except RuntimeError as e:
            log(f"  Metal bake failed: {e}")
            log("  Falling back to KDTree texture baker...")

    log(f"Baking PBR textures via KDTree ({tex_size}x{tex_size})...")
    from backends.texture_baker import uv_unwrap, bake_texture, export_glb_with_texture

    voxel_coords = mesh_out.coords.cpu().float().numpy()
    voxel_attrs = mesh_out.attrs.cpu().float().numpy()
    origin = mesh_out.origin.cpu().float().numpy()
    vs = mesh_out.voxel_size

    # Simplify before UV unwrap — xatlas is very slow on 800K+ vertex meshes.
    bake_verts, bake_faces = verts, faces
    target_faces = min(BAKE_TARGET_FACES, len(faces))
    if len(faces) > target_faces:
        try:
            import fast_simplification

            ratio = 1.0 - (target_faces / len(faces))
            log(f"  Simplifying mesh: {len(faces):,} -> ~{target_faces:,} faces")
            bake_verts, bake_faces = fast_simplification.simplify(verts, faces, ratio)
        except ImportError:
            log("  Warning: fast_simplification not installed, UV unwrapping full mesh (slow)")

    log("  UV unwrapping with xatlas...")
    new_verts, new_faces, uvs, _vmapping = uv_unwrap(bake_verts, bake_faces)
    log(f"  UV unwrap: {len(verts):,} -> {len(new_verts):,} vertices")

    base_color_img, mr_img, _mask = bake_texture(
        new_verts, new_faces, uvs,
        voxel_coords, voxel_attrs, origin, vs,
        texture_size=tex_size,
    )

    basecolor_path = f"{output}_basecolor.png"
    PILImage.fromarray(base_color_img).save(basecolor_path)
    export_glb_with_texture(new_verts, new_faces, uvs, base_color_img, mr_img, glb_path)
    log(f"  Saved: {glb_path}")
    return basecolor_path
