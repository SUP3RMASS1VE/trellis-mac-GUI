"""Pydantic request/response models for the Trellis2 API server."""
from typing import Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request to generate a 3D model from an image."""
    image: str = Field(..., description="Base64-encoded image (PNG/JPEG)")
    seed: int = Field(default=42, description="Random seed")
    pipeline_type: str = Field(
        default="1024_cascade",
        description="Pipeline type: 512, 1024, 1024_cascade, 1536_cascade",
    )
    output_path: Optional[str] = Field(default=None, description="Save GLB to this path (in addition to returning base64)")
    decimation_target: int = Field(default=1000000, description="Target face count for simplification")
    texture_size: int = Field(default=2048, description="Texture resolution")
    remesh: bool = Field(default=False, description="Whether to remesh the output")
    steps: Optional[int] = Field(default=None, description="Number of sampler steps (default: 12, lower = faster)")
    guidance_strength: Optional[float] = Field(default=None, description="Guidance strength for structure/shape samplers (default: 7.5)")
    texture_guidance: Optional[float] = Field(default=None, description="Guidance strength for texture sampler (default: 1.0 = OFF)")


class GenerateResponse(BaseModel):
    """Response containing the generated 3D model."""
    glb: str = Field(..., description="Base64-encoded GLB file")
    vertices: int = Field(..., description="Number of vertices in the output mesh")
    faces: int = Field(..., description="Number of faces in the output mesh")
    generation_time: float = Field(..., description="Generation time in seconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    backend: str = ""
    weights_loaded: bool = False
