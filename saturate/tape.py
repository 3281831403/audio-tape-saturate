"""Tape-saturation DSP primitives for the audio-tape-saturate exercise.

Audio is plain ``list[float]`` in the range ``[-1.0, 1.0]`` unless a function
says otherwise.  Every function below is a stub and raises
``NotImplementedError``; the docstrings define the contract that
``tests/test_tape.py`` checks.  Unless stated otherwise, numeric comparisons
use an absolute tolerance of ``1e-6``.
"""

from __future__ import annotations

import math

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


class TapeError(Exception):
    """Raised when a parameter or an input track violates the contract."""


def soft_clip(x: float, drive: float) -> float:
    """Apply a memoryless tanh-style soft clipper to one sample.

    Contract:
        * ``drive`` is a positive input gain.  ``drive <= 0`` raises
          ``TapeError``; the curve is normalised by ``tanh(drive)`` so that a
          full-scale input still maps to a full-scale output.
        * The transfer curve is odd and smooth: ``f(-x) == -f(x)`` and
          ``f(0.0) == 0.0``.  It compresses instead of hard-clipping, so a
          half-scale input lands well above ``0.5`` for a large drive.
        * Output is bounded by ``[-1.0, 1.0]`` for ``x`` in ``[-1.0, 1.0]``.
          Raising ``drive`` increases compression; it must not scale the
          output level linearly.
        * Absolute tolerance for comparisons: ``1e-6``.
    """
    raise NotImplementedError


def harmonics(x: float, amount: float) -> list[float]:
    """Return the 2nd and 3rd harmonic contributions of a single sample.

    Contract:
        * ``amount`` is the harmonic amount in ``[0.0, 1.0]``; values outside
          that range raise ``TapeError``.
        * The result is the two-element list ``[second, third]``.  Both
          entries scale linearly with ``amount``: ``amount == 0.0`` returns
          ``[0.0, 0.0]`` and ``amount == 1.0`` returns the full contribution.
          At ``x == 0.0`` the result is ``[0.0, 0.0]`` as well.
        * Absolute tolerance for comparisons: ``1e-6``.
    """
    raise NotImplementedError


def dc_block(samples: list[float], cutoff: float) -> list[float]:
    """Remove DC with a causal first-order high-pass filter.

    Contract:
        * ``cutoff`` is the normalised corner frequency ``f_c / sample_rate``
          and must be in ``(0.0, 0.5)``; a non-positive or too-large value
          raises ``TapeError`` instead of dividing by zero.
        * The filter is stable, so a constant input settles to ``0.0`` in
          steady state (the mean of the tail is within ``1e-6`` of zero).
        * Content above the corner is preserved: a sine at ten times the
          corner keeps its amplitude within 10 %, while a DC offset is
          removed.  Empty input returns ``[]``.
        * Absolute tolerance for comparisons: ``1e-6``.
    """
    raise NotImplementedError


def saturate(samples: list[float], drive: float, mix: float) -> list[float]:
    """Saturate a whole track with a dry/wet blend.

    Contract:
        * ``samples`` are in ``[-1.0, 1.0]``; empty input returns ``[]``.
        * ``drive > 0`` and ``0.0 <= mix <= 1.0``; any other value raises
          ``TapeError``.
        * The output is ``(1.0 - mix) * dry + mix * wet`` where ``wet`` is
          the ``soft_clip`` of the sample.  The weights sum to 1.0, so a quiet
          signal is not made louder at an intermediate ``mix``; ``mix == 0.0``
          returns the input unchanged (``1e-9`` tolerance).
        * Every output sample is bounded by ``[-1.0, 1.0]``.
        * Absolute tolerance for comparisons: ``1e-6``.
    """
    raise NotImplementedError


def tape_wobble(
    samples: list[float], depth: float, rate: float, sample_rate: float
) -> list[float]:
    """Apply deterministic wow/flutter delay modulation to a track.

    Contract:
        * ``depth`` is in ``[0.0, 1.0]``, ``rate > 0`` (Hz) and
          ``sample_rate > 0`` (Hz); invalid values raise ``TapeError``.
        * Sample ``n`` is read at position ``n - delay_n`` (linear
          interpolation), with
          ``delay_n = sample_rate * depth * 0.001 * sin(2*pi*rate*n/sample_rate)``.
          The peak delay is therefore ``depth`` milliseconds, and reads
          outside ``[0, len(samples) - 1]`` count as ``0.0``.
        * The LFO phase comes from time ``n / sample_rate``, so the
          modulation frequency in Hz is independent of ``sample_rate``: on a
          linear ramp, ``output[n] - input[n]`` is that delayed sine in
          samples, and its period in seconds is ``1 / rate``.
        * Deterministic: no randomness and no sleeping, same input gives the
          same output.
        * Absolute tolerance for comparisons: ``1e-6``.
    """
    raise NotImplementedError


def hysteresis(x: float, prev: float, width: float) -> float:
    """Advance a magnetic hysteresis state by one input sample.

    Contract:
        * ``width`` is clamped into ``[0.0, 1.0]`` before use.  A negative
          width therefore behaves like ``0.0`` and must never move the state
          against the input.
        * The output starts at ``prev`` and moves toward ``x`` by at most
          ``width`` (never overshooting ``x``).  For a monotonically
          increasing input the output is monotonically non-decreasing.
        * ``width == 0.0`` leaves ``prev`` unchanged.
        * Absolute tolerance for comparisons: ``1e-6``.
    """
    raise NotImplementedError


def measure_thd(samples: list[float], freq: float, sample_rate: float) -> float:
    """Measure total harmonic distortion (THD) as an energy ratio.

    Contract:
        * ``freq > 0`` and ``sample_rate > freq``; invalid values raise
          ``TapeError``.
        * The fundamental sits at ``freq`` and the harmonics at
          ``2*freq, 3*freq, ...`` below Nyquist.  THD is the RMS energy of the
          harmonics divided by the RMS energy of the fundamental, not a ratio
          of peak values.
        * A pure sine at ``freq`` (integer number of cycles) returns ``0.0``
          within ``1e-6``.  A sine plus a 3rd harmonic 10 % as large returns
          ``0.1`` within ``1e-3``.  The result is clamped to ``[0.0, 1.0]``.
        * Empty input or a zero fundamental returns ``0.0``.
        * Absolute tolerance for comparisons: ``1e-6``.
    """
    raise NotImplementedError


def saturate_batch(
    tracks: dict[str, list[float]], config: dict
) -> dict[str, list[float]]:
    """Saturate several tracks in one call and return a new dict.

    Contract:
        * ``tracks`` maps a track name to ``list[float]`` audio.  ``config``
          carries ``drive`` (required), ``mix`` (default ``1.0``) and
          ``width`` (hysteresis width, default ``0.0``).
        * Each track is processed by ``saturate`` and then, sample by sample,
          by ``hysteresis`` starting from its own fresh ``prev = 0.0``.
          Hysteresis state must never leak from one track to another.
        * Validation is all-or-nothing: if any track or config value is
          invalid the call raises ``TapeError`` and returns nothing, so a
          failed batch cannot expose partially processed audio (rollback).
        * The returned dict keeps the insertion order of ``tracks``; on an
          all-success run its keys therefore match the input order exactly.
        * Absolute tolerance for comparisons: ``1e-6``.
    """
    raise NotImplementedError
