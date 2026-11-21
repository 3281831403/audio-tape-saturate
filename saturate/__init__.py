"""Audio tape saturation exercise package."""

from .tape import (
    TapeError,
    dc_block,
    harmonics,
    hysteresis,
    measure_thd,
    saturate,
    saturate_batch,
    soft_clip,
    tape_wobble,
)

__all__ = [
    "TapeError",
    "soft_clip",
    "harmonics",
    "dc_block",
    "saturate",
    "tape_wobble",
    "hysteresis",
    "measure_thd",
    "saturate_batch",
]
