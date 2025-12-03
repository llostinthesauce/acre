from __future__ import annotations

import types
import sys
from typing import Any

try:
    from packaging.version import Version
except Exception:
    Version = None  # type: ignore[assignment]


def _torch_version_ge(version: str) -> bool:
    try:
        import torch

        if Version:
            return Version(torch.__version__) >= Version(version)
        return torch.__version__.split(".") >= version.split(".")
    except Exception:
        return False


def ensure_tensor_parallel_stub() -> None:
    """
    When torch < 2.5, ``transformers`` tries to import DTensor from
    ``transformers.integrations.tensor_parallel`` and fails. This helper patches
    the module with minimal stubs before Transformers imports it.
    """
    name = "transformers.integrations.tensor_parallel"
    force_stub = not _torch_version_ge("2.5")
    if not force_stub:
        return

    # Always (re)install a stub to satisfy imports that expect newer APIs.
    module = types.ModuleType(name)

    class DummyLayer:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

    def _no_op(*args: Any, **kwargs: Any):
        return args[0] if args else None

    module.ALL_PARALLEL_STYLES = []
    module.DTensor = object
    module.Replicate = DummyLayer
    module.TensorParallelLayer = DummyLayer
    module.Placement = object
    module.Shard = object
    module._get_parameter_tp_plan = lambda *args, **kwargs: None
    module.distribute_model = _no_op
    module.repack_weights = _no_op
    module.replace_state_dict_local_with_dtensor = _no_op
    module.verify_tp_plan = _no_op
    module.initialize_tensor_parallelism = lambda *args, **kwargs: (None, None, 1)
    module.shard_and_distribute_module = _no_op
    module.translate_to_torch_parallel_style = _no_op

    sys.modules[name] = module
