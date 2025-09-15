from __future__ import annotations

from typing import Any

import types


def device_ptr_capsule(ptr: int) -> Any:
    """Create a PyCapsule that holds a raw device pointer.

    Many frameworks (e.g., PyTorch) provide an integer pointer via `.data_ptr()`.
    This helper wraps that integer as a capsule consumable by the native copy engine.
    """
    import ctypes
    import sys

    name = b"device_ptr"
    # Create a capsule using CPython C-API via ctypes
    # PyCapsule* PyCapsule_New(void *pointer, const char *name, PyCapsule_Destructor destructor)
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return PyCapsule_New(ctypes.c_void_p(ptr), name, None)


def from_torch_tensor(tensor: Any) -> Any:
    """Return a device pointer capsule for a torch.Tensor (CUDA/HIP) without importing torch here.

    The caller should have a torch tensor; we avoid hard dependency to keep the package light.
    """
    ptr = int(getattr(tensor, "data_ptr")())
    return device_ptr_capsule(ptr)

