from typing import *
import platform
import sys

CONV = 'flex_gemm'
DEBUG = False
ATTN = 'flash_attn'

def __detect_defaults():
    """Auto-detect best backends for current platform."""
    global CONV, ATTN
    if platform.system() == 'Darwin':
        if __flex_gemm_works_on_mps():
            CONV = 'flex_gemm'
            # flex_gemm_sparse_attn is now flash-attention-v2 with simdgroup
            # matmul + simd-shuffle softmax (5–15× over SDPA-padded-CPU-bounce
            # at every measured shape including max_seqlen=2048). The conv
            # probe above covers the same package, so if it passed, the
            # attention path is available too.
            ATTN = 'flex_gemm_sparse_attn'
        else:
            CONV = 'pytorch'
            ATTN = 'sdpa'
    elif not __has_cuda():
        CONV = 'pytorch'
        ATTN = 'sdpa'


def __flex_gemm_works_on_mps():
    """Probe flex_gemm with tiny MPS convs covering both IMPLICIT_GEMM and
    MASKED_IMPLICIT_GEMM. If the install pre-dates the device-routing fix
    (or pre-dates the real masked kernel in round 2), one of these returns
    a CPU tensor — fall back to the pure-PyTorch backend rather than
    crashing inside the model on the first LayerNorm. Build tensors on CPU
    and move to MPS because some PyTorch builds lack int/fp16 torch.zeros
    kernels on MPS."""
    try:
        import torch
        if not torch.backends.mps.is_available():
            return False
        import flex_gemm
        from flex_gemm.ops.spconv import sparse_submanifold_conv3d, Algorithm, set_algorithm

        # Exercise both algorithms — masked carries its own cache/dispatch path
        # distinct from dense. A stale install may have one working and the
        # other broken (e.g. the pre-round-2 aliased-to-dense fallback).
        coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32).to('mps')
        feats = torch.zeros((1, 4), dtype=torch.float16).to('mps')
        weight = torch.zeros((4, 1, 1, 1, 4), dtype=torch.float16).to('mps')
        shape = torch.Size([1, 4, 1, 1, 1])

        for algo in (Algorithm.IMPLICIT_GEMM, Algorithm.MASKED_IMPLICIT_GEMM):
            set_algorithm(algo)
            out, _ = sparse_submanifold_conv3d(feats, coords, shape, weight)
            if out.device.type != 'mps':
                return False
        return True
    except Exception:
        return False

def __has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def __from_env():
    import os

    global CONV
    global DEBUG
    global ATTN

    __detect_defaults()

    env_sparse_conv_backend = os.environ.get('SPARSE_CONV_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn_backend = os.environ.get('SPARSE_ATTN_BACKEND')
    if env_sparse_attn_backend is None:
        env_sparse_attn_backend = os.environ.get('ATTN_BACKEND')

    if env_sparse_conv_backend is not None and env_sparse_conv_backend in ['none', 'spconv', 'torchsparse', 'flex_gemm', 'pytorch']:
        CONV = env_sparse_conv_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    if env_sparse_attn_backend is not None and env_sparse_attn_backend in [
        'xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'flex_gemm_sparse_attn',
    ]:
        ATTN = env_sparse_attn_backend

    print(f"[SPARSE] Conv backend: {CONV}; Attention backend: {ATTN}")


__from_env()


def set_conv_backend(backend: Literal['none', 'spconv', 'torchsparse', 'flex_gemm', 'pytorch']):
    global CONV
    CONV = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

def set_attn_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'flex_gemm_sparse_attn']):
    global ATTN
    ATTN = backend
