"""Tape-saturation DSP primitives for the audio-tape-saturate exercise.

Audio is plain ``list[float]`` in the range ``[-1.0, 1.0]`` unless a function
says otherwise.  The contract each function implements is defined by its
docstring and checked by ``tests/test_tape.py``.  Unless stated otherwise,
numeric comparisons use an absolute tolerance of ``1e-6``.
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
    if not drive > 0.0:
        raise TapeError("drive must be positive")

    # Normalising by tanh(drive) keeps a full-scale input at full scale, so
    # raising the drive adds compression instead of linear gain.
    return math.tanh(drive * x) / math.tanh(drive)


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
    if not 0.0 <= amount <= 1.0:
        raise TapeError("amount must be in [0.0, 1.0]")

    # Even (x**2) and odd (x**3) distortion terms, both proportional to the
    # requested amount so that 0 gives no harmonics and 1 gives the full set.
    second = amount * 0.5 * x * x
    third = amount * 0.25 * x * x * x
    return [second, third]


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
    if not 0.0 < cutoff < 0.5:
        raise TapeError("cutoff must be in (0.0, 0.5)")

    if not samples:
        return []

    # y[n] = x[n] - x[n-1] + R * y[n-1].  R = exp(-2*pi*cutoff) keeps the pole
    # inside the unit circle for every valid cutoff and places the corner at
    # the requested normalised frequency; the zero at DC forces steady state
    # to 0 while content above the corner passes through.
    pole = math.exp(-2.0 * math.pi * cutoff)
    output: list[float] = []
    previous_input = 0.0
    previous_output = 0.0
    for value in samples:
        current = value - previous_input + pole * previous_output
        output.append(current)
        previous_input = value
        previous_output = current
    return output


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
    if not drive > 0.0:
        raise TapeError("drive must be positive")
    if not 0.0 <= mix <= 1.0:
        raise TapeError("mix must be in [0.0, 1.0]")

    if not samples:
        return []

    dry_gain = 1.0 - mix
    return [
        dry_gain * value + mix * soft_clip(value, drive) for value in samples
    ]


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
    if not 0.0 <= depth <= 1.0:
        raise TapeError("depth must be in [0.0, 1.0]")
    if not rate > 0.0:
        raise TapeError("rate must be positive")
    if not sample_rate > 0.0:
        raise TapeError("sample_rate must be positive")

    if not samples:
        return []

    last = len(samples) - 1
    output: list[float] = []
    # The LFO phase advances by rate / sample_rate cycles per sample, i.e. it
    # is driven by elapsed time rather than the frame index, so the
    # modulation frequency in Hz does not depend on the sample rate.
    phase_step = 2.0 * math.pi * rate / sample_rate
    phase = 0.0
    for index, _ in enumerate(samples):
        delay = sample_rate * depth * 0.001 * math.sin(phase)
        # Wrap the phase so precision does not degrade on long tracks.
        phase = (phase + phase_step) % (2.0 * math.pi)
        position = index - delay
        if position < 0.0 or position > last:
            output.append(0.0)
            continue

        left = int(math.floor(position))
        fraction = position - left
        if left >= last:
            output.append(samples[last])
        else:
            output.append(
                samples[left] * (1.0 - fraction) + samples[left + 1] * fraction
            )
    return output


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
    # Clamping keeps a negative width from producing a state step that runs
    # opposite to the input (which is what made the input appear to reverse).
    clamped = min(max(width, 0.0), 1.0)
    step = x - prev
    if step > clamped:
        step = clamped
    elif step < -clamped:
        step = -clamped
    return prev + step


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
    if not freq > 0.0:
        raise TapeError("freq must be positive")
    if not sample_rate > freq:
        raise TapeError("sample_rate must be greater than freq")

    count = len(samples)
    if count == 0:
        return 0.0

    def amplitude(multiple: float) -> float:
        """Single-bin DFT amplitude at ``multiple * freq``."""
        omega = 2.0 * math.pi * multiple * freq / sample_rate
        real = 0.0
        imaginary = 0.0
        for index, value in enumerate(samples):
            angle = omega * index
            real += value * math.cos(angle)
            imaginary += value * math.sin(angle)
        return math.hypot(2.0 * real / count, 2.0 * imaginary / count)

    fundamental = amplitude(1.0)
    if fundamental == 0.0:
        return 0.0

    # Sum the *energy* (squared amplitudes) of every harmonic below Nyquist.
    harmonic_energy = 0.0
    multiple = 2.0
    while multiple * freq < sample_rate / 2.0:
        value = amplitude(multiple)
        harmonic_energy += value * value
        multiple += 1.0

    thd = math.sqrt(harmonic_energy) / fundamental
    return min(max(thd, 0.0), 1.0)


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
    if not isinstance(config, dict):
        raise TapeError("config must be a dict")
    if "drive" not in config:
        raise TapeError("config requires a drive value")

    drive = config["drive"]
    mix = config.get("mix", 1.0)
    width = config.get("width", 0.0)

    if not isinstance(drive, (int, float)) or isinstance(drive, bool):
        raise TapeError("drive must be a number")
    if not drive > 0.0:
        raise TapeError("drive must be positive")
    if not isinstance(mix, (int, float)) or isinstance(mix, bool):
        raise TapeError("mix must be a number")
    if not 0.0 <= mix <= 1.0:
        raise TapeError("mix must be in [0.0, 1.0]")
    if not isinstance(width, (int, float)) or isinstance(width, bool):
        raise TapeError("width must be a number")

    if not isinstance(tracks, dict):
        raise TapeError("tracks must be a dict")

    # Validate every track up front so a bad track aborts the whole batch
    # before any audio is processed (all-or-nothing rollback).
    for name, track in tracks.items():
        if not isinstance(track, (list, tuple)):
            raise TapeError(f"track {name!r} must be a sequence")
        for value in track:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TapeError(f"track {name!r} contains a non-numeric sample")
            if not math.isfinite(value):
                raise TapeError(f"track {name!r} contains a non-finite sample")

    output: dict[str, list[float]] = {}
    for name, track in tracks.items():
        wet = saturate(list(track), drive, mix)
        # A fresh state per track: hysteresis must never bleed across tracks.
        previous = 0.0
        processed: list[float] = []
        for value in wet:
            previous = hysteresis(value, previous, width)
            processed.append(previous)
        output[name] = processed
    return output
