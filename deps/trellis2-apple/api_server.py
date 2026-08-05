"""
FastAPI server for Trellis2 image-to-3D generation (MLX backend).

- POST /generate — base64 image → GLB
- GET /health — server status

Usage:
    python api_server.py --weights weights/TRELLIS.2-4B --port 8082
"""
import os
import sys

# Environment defaults — set before any torch/trellis imports
os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # deform_conv2d in RMBG

# Add o-voxel to path if present
_ovoxel = os.path.join(os.path.dirname(os.path.abspath(__file__)), "o-voxel")
if os.path.isdir(_ovoxel) and _ovoxel not in sys.path:
    sys.path.insert(0, _ovoxel)
import io
import base64
import time
import tempfile
import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image

from api_models import GenerateRequest, GenerateResponse, HealthResponse
from mlx_backend import setup_logging

# Global pipeline instance
pipeline = None


@asynccontextmanager
async def lifespan(app):
    global pipeline
    from mlx_backend.pipeline import create_mlx_pipeline

    weights = os.environ.get("TRELLIS2_WEIGHTS", "weights/TRELLIS.2-4B")
    pipeline = create_mlx_pipeline(weights_path=weights)
    yield
    pipeline = None


app = FastAPI(title="Trellis2 MLX API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    if pipeline is None:
        return HealthResponse(status="loading")
    return HealthResponse(
        status="ok",
        backend="mlx",
        weights_loaded=True,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    t_start = time.time()

    # Decode input image
    try:
        image_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Build per-sampler overrides (upstream defaults: structure/shape=7.5, texture=1.0)
    structure_shape_overrides = {}
    if request.steps is not None:
        structure_shape_overrides['steps'] = request.steps
    if request.guidance_strength is not None:
        structure_shape_overrides['guidance_strength'] = request.guidance_strength

    texture_overrides = {}
    if request.steps is not None:
        texture_overrides['steps'] = request.steps
    if request.texture_guidance is not None:
        texture_overrides['guidance_strength'] = request.texture_guidance

    # Generate mesh
    try:
        meshes = pipeline.run(
            image,
            seed=request.seed,
            pipeline_type=request.pipeline_type,
            sparse_structure_sampler_params=structure_shape_overrides,
            shape_slat_sampler_params=structure_shape_overrides,
            tex_slat_sampler_params=texture_overrides,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    mesh = meshes[0]

    # Export to GLB
    try:
        if request.output_path:
            out_dir = os.path.dirname(request.output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            glb_path = request.output_path
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
            glb_path = tmp.name
            tmp.close()

        from mlx_backend.pipeline import to_glb
        to_glb(
            mesh, glb_path,
            decimation_target=request.decimation_target,
            texture_size=request.texture_size,
            remesh=request.remesh,
            verbose=True,
        )
        with open(glb_path, "rb") as f:
            glb_bytes = f.read()
        if not request.output_path:
            os.unlink(glb_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"GLB export failed: {e}")

    t_end = time.time()

    return GenerateResponse(
        glb=base64.b64encode(glb_bytes).decode(),
        vertices=int(mesh.vertices.shape[0]),
        faces=int(mesh.faces.shape[0]),
        generation_time=round(t_end - t_start, 2),
    )


def main():
    parser = argparse.ArgumentParser(description="Trellis2 MLX API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--weights", type=str, default="weights/TRELLIS.2-4B")
    args = parser.parse_args()

    os.environ["TRELLIS2_WEIGHTS"] = args.weights
    setup_logging()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
