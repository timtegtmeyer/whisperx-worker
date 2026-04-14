# cog_stub.py  ───────────────────────────────────────────────────────────
from pathlib import Path as _Path

# ----------------------------------------------------
# Minimal stand-ins for the 4 cog symbols we import
# ----------------------------------------------------
class Input:                         # noqa: N801
    """Stand-in for cog.Input — returns the default value so it works as a
    function-parameter default: ``beam_size: int = Input(default=8)``."""
    def __new__(cls, *_args, default=None, **_kwargs):
        return default

class BasePredictor:                 # noqa: N801
    def setup(self):
        pass

Path = _Path                         # noqa: N801

class BaseModel:                     # noqa: N801
    """Accept any keyword args and store them as attributes."""
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)

    def dict(self):
        return self.__dict__
# -----------------------------------------------------------------------