from __future__ import annotations

import importlib
from typing import Any, Callable


def load_callable(spec: str) -> Callable[..., Any]:
    """Load a callable from a `module:function` spec string.

    Example: `mypkg.mymodule:myfunc`
    """
    if ":" not in spec:
        raise ValueError("spec must be 'module:function'")
    mod_name, func_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn

