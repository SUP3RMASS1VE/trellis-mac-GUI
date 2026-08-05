# Vendored sources

These trees ship inside this repository so `setup.sh` needs no cloning. Recorded
here are the upstream origins and the exact commits they were vendored from.

| Tree | Upstream | Commit | Python module |
|---|---|---|---|
| `../TRELLIS.2` | https://github.com/microsoft/TRELLIS.2 | `75fbf01` | `trellis2` |
| `utils3d` | https://github.com/EasternJournalist/utils3d | `9a4eb15` | `utils3d` |
| `mtlbvh` | https://github.com/pedronaugusto/mtlbvh | `6b2a0f6` | `mtlbvh` |
| `mtldiffrast` | https://github.com/pedronaugusto/mtldiffrast | `c9499ba` | `mtldiffrast` |
| `mtlmesh` | https://github.com/pedronaugusto/mtlmesh | `b63fd24` | `cumesh` |
| `mtlgemm` | https://github.com/pedronaugusto/mtlgemm | `b75a52d` | `flex_gemm` |
| `trellis2-apple` | https://github.com/pedronaugusto/trellis2-apple | `6055b86` | `o_voxel` |

## TRELLIS.2 is vendored pre-patched

`TRELLIS.2/` is **not** pristine upstream — 9 files carry the MPS compatibility
changes applied by `patches/mps_compat.py`. The diff against upstream `75fbf01`
is recorded in `patches/vendored-trellis2.diff`.

`setup.sh` still runs `patches/mps_compat.py`; every patch is guarded, so on a
fresh clone it reports "Already patched" instead of applying anything twice.

## Updating a vendored tree

```bash
# Example: refresh mtlgemm from upstream
rm -rf deps/mtlgemm
git clone https://github.com/pedronaugusto/mtlgemm.git deps/mtlgemm
rm -rf deps/mtlgemm/.git          # keep it a plain directory, not a nested repo
bash setup.sh                     # rebuilds the Metal extension
```

Nested `.git` directories must not be reintroduced: git stores a directory
containing its own `.git` as a gitlink (mode 160000) and skips the files inside,
so the tree would clone empty for everyone else.

The original clone metadata for each tree was renamed to `.git.vendored` rather
than deleted, and is gitignored. To restore upstream history in a tree locally:

```bash
mv deps/mtlgemm/.git.vendored deps/mtlgemm/.git
```
