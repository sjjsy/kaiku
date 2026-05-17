"""Minimal import check for release wheels (run via uv publish workflow)."""

import kaiku
from kaiku.config import _CONFIG_TEMPLATE

assert kaiku.__version__
assert "kaiku configuration" in _CONFIG_TEMPLATE
