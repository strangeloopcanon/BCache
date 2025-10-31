from __future__ import annotations

import sys
import types

from bodocache.integrations.loader import load_callable


def test_load_callable_dynamic():
    mod = types.ModuleType("test_plugin_mod")

    def f(x):
        return x + 1

    mod.f = f
    sys.modules["test_plugin_mod"] = mod
    fn = load_callable("test_plugin_mod:f")
    assert fn(41) == 42
