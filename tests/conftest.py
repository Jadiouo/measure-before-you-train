import ast
import os
import types

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import animalbench  # noqa: F401  (enables x64)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_OUT = os.path.join(ROOT, "calib", "out")
os.makedirs(CALIB_OUT, exist_ok=True)


@pytest.fixture(scope="session")
def poc():
    """Load wake/power reference functions from poc/wake_rotation_poc.py WITHOUT importing the module
    (importing it opens/truncates poc/out/log.txt)."""
    src = open(os.path.join(ROOT, "poc", "wake_rotation_poc.py")).read()
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("powers", "wbar", "save_frac", "calibrate"):
            keep.append(node)
        elif isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in ("G", "BASE"):
            keep.append(node)
    mod = ast.Module(body=keep, type_ignores=[])
    ns = {"np": np}
    exec(compile(mod, "poc_ref", "exec"), ns)
    return types.SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})
