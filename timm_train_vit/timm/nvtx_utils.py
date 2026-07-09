"""Toggleable NVTX ranges — the Python-side equivalent of the C++ -DUSE_NVTX build.

Enabled with USE_NVTX=1 in the environment. When unset, nvtx_range is a no-op
context manager, so the normal (non-profiling) run_py.sh path pays zero cost.
"""
import os
from contextlib import contextmanager, nullcontext

USE_NVTX = os.environ.get("USE_NVTX") == "1"

if USE_NVTX:
    import torch

    @contextmanager
    def nvtx_range(name):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
else:
    def nvtx_range(name):
        return nullcontext()
