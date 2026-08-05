"""
Centralized backend resolution for trellis2-apple.

Resolves differentiable rasterization (dr), mesh processing (MeshBackend, BVH),
remeshing, and grid_sample once at import time. All other modules import from here.

Priority: Metal (mtldiffrast) > CUDA (nvdiffrast) > CPU fallback.
Mesh: cumesh (auto-selects Metal/CUDA internally).
"""
import platform
import torch

# ---------------------------------------------------------------------------
# Detect platform
# ---------------------------------------------------------------------------
HAS_MPS = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
HAS_CUDA = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# Differentiable rasterizer  (dr)
# mtldiffrast (Metal) and nvdiffrast (CUDA) are separate packages
# ---------------------------------------------------------------------------
dr = None
_dr_backend = None

try:
    import mtldiffrast.torch as _mtldr
    dr = _mtldr
    _dr_backend = 'metal'
except ImportError:
    pass

if dr is None:
    try:
        import nvdiffrast.torch as _nvdr
        dr = _nvdr
        _dr_backend = 'cuda'
    except ImportError:
        pass

HAS_DR = dr is not None


def RasterizeContext(device=None):
    """Create the appropriate rasterization context for the active backend."""
    if _dr_backend == 'metal':
        return dr.MtlRasterizeContext(device=device)
    elif _dr_backend == 'cuda':
        return dr.RasterizeCudaContext(device=device)
    raise RuntimeError("No differentiable rasterization backend available")

# ---------------------------------------------------------------------------
# Mesh processing  (MeshBackend, BVH, remesh)
# cumesh auto-selects Metal or CUDA backend internally
# ---------------------------------------------------------------------------
MeshBackend = None
BVH = None
remesh_narrow_band_dc = None
_mesh_backend = None

try:
    import cumesh
    MeshBackend = cumesh.CuMesh
    BVH = cumesh.cuBVH
    remesh_narrow_band_dc = cumesh.remeshing.remesh_narrow_band_dc
    _mesh_backend = 'metal' if platform.system() == 'Darwin' else 'cuda'
except ImportError:
    pass

HAS_MESH = MeshBackend is not None
HAS_REMESH = remesh_narrow_band_dc is not None

# ---------------------------------------------------------------------------
# CPU fallback mesh libraries
# ---------------------------------------------------------------------------
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    trimesh = None
    HAS_TRIMESH = False

try:
    import fast_simplification
    HAS_FAST_SIMPLIFICATION = True
except ImportError:
    fast_simplification = None
    HAS_FAST_SIMPLIFICATION = False

# ---------------------------------------------------------------------------
# Grid sample (flex_gemm on Metal/CUDA, F.grid_sample fallback)
# ---------------------------------------------------------------------------
_flex_grid_sample_3d = None
HAS_FLEX_GEMM = False

try:
    from flex_gemm.ops.grid_sample import grid_sample_3d as _fgs3d
    _flex_grid_sample_3d = _fgs3d
    HAS_FLEX_GEMM = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Overall backend name
# ---------------------------------------------------------------------------
BACKEND = _dr_backend or _mesh_backend or ('cpu' if HAS_TRIMESH else None)
HAS_GPU = HAS_MPS or HAS_CUDA
