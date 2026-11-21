"""Red tests for the audio-tape-saturate exercise.

Fifteen tests, one per known defect plus five contract/regression checks.
Every test calls at least one stub in ``saturate.tape``; on the initial
repository all fifteen fail with ``NotImplementedError``.  Do not edit this
file: fix ``saturate/tape.py`` instead.
"""

import math

import pytest

from saturate.tape import (
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

# Defect 1: soft clipping is implemented as a hard clip, so the harmonics are
# far too strong around half scale.


def test_soft_clip_is_smooth_not_hard_clipped():
    low = soft_clip(0.25, 6.0)
    mid = soft_clip(0.5, 6.0)
    high = soft_clip(1.0, 6.0)

    assert soft_clip(0.0, 6.0) == pytest.approx(0.0, abs=1e-6)
    assert soft_clip(-0.5, 6.0) == pytest.approx(-mid, abs=1e-6)
    assert 0.25 < low < mid < high <= 1.0 + 1e-6
    # A hard clip would return exactly 0.5 here, and 0.25 below it.
    assert mid > 0.9
    assert mid / low < 1.5


# Defect 2: the drive is not normalised, so the output gets linearly louder.


def test_drive_is_normalised_and_not_linear_gain():
    quiet_low = soft_clip(0.5, 2.0)
    quiet_high = soft_clip(0.5, 8.0)

    assert 0.0 < quiet_low < quiet_high <= 1.0 + 1e-6
    # An unnormalised drive would multiply the level by four here.
    assert quiet_high / quiet_low < 2.0
    assert soft_clip(1.0, 8.0) <= 1.0 + 1e-6


# Defect 3: the harmonic generator ignores ``amount`` and always uses 50 %.


def test_harmonics_scale_with_amount():
    silent = harmonics(0.8, 0.0)
    quarter = harmonics(0.8, 0.25)
    full = harmonics(0.8, 1.0)

    assert len(silent) == 2
    assert silent == pytest.approx([0.0, 0.0], abs=1e-6)
    assert full == pytest.approx([value * 4.0 for value in quarter], abs=1e-6)
    assert any(abs(value) > 1e-6 for value in full)


# Defect 4: the DC blocker has no steady state and eats low frequencies.


def test_dc_block_removes_dc_and_keeps_low_frequencies():
    sample_rate = 1000
    count = 2000
    cutoff = 0.01

    dc_input = [0.5] * count
    dc_output = dc_block(dc_input, cutoff)
    assert abs(sum(dc_output[-200:]) / 200.0) < 1e-6

    tone = [
        math.sin(2.0 * math.pi * 0.1 * index) for index in range(count)
    ]
    tone_output = dc_block(tone, cutoff)
    input_rms = math.sqrt(sum(value * value for value in tone[-400:]) / 400.0)
    output_rms = math.sqrt(
        sum(value * value for value in tone_output[-400:]) / 400.0
    )
    assert output_rms / input_rms > 0.9


# Defect 5: the dry/wet weights do not sum to 1, so the whole mix gets louder.


def test_dry_wet_mix_has_unity_gain():
    samples = [0.5, -0.25, 0.125, -0.75]
    # A very small drive keeps the soft clipper close to the identity.
    mixed = saturate(samples, 1e-3, 0.5)
    assert mixed == pytest.approx(samples, abs=1e-6)


# Defect 6: the wobble phase uses the frame index, so changing the sample
# rate changes the modulation pitch.


def _modulation_rate(sample_rate):
    rate = 5.0
    depth = 0.5
    count = int(sample_rate * 0.5)
    ramp = [float(index) for index in range(count)]
    wobbled = tape_wobble(ramp, depth, rate, sample_rate)
    difference = [wobbled[index] - ramp[index] for index in range(count)]

    low = int(0.05 * sample_rate)
    high = int(0.45 * sample_rate)
    window = difference[low:high]
    crossings = sum(
        1 for left, right in zip(window, window[1:]) if left * right < 0.0
    )
    duration = (high - low) / sample_rate
    return crossings / (2.0 * duration)


def test_wobble_modulation_rate_is_sample_rate_independent():
    rate_8k = _modulation_rate(8000)
    rate_16k = _modulation_rate(16000)

    assert rate_8k == pytest.approx(5.0, abs=0.6)
    assert rate_16k == pytest.approx(5.0, abs=0.6)
    assert rate_8k == pytest.approx(rate_16k, abs=0.6)


# Defect 7: the hysteresis width is not clamped, so the output can reverse.


def test_hysteresis_clamps_width_and_never_reverses():
    previous = 0.0
    for sample in (0.05, 0.15, 0.3, 0.45, 0.6):
        current = hysteresis(sample, previous, -0.5)
        assert current >= previous - 1e-6
        previous = current

    previous = 0.0
    for sample in (0.2, 0.4, 0.6, 0.8, 1.0):
        current = hysteresis(sample, previous, 0.25)
        assert previous - 1e-6 <= current <= sample + 1e-6
        assert current - previous <= 0.25 + 1e-6
        previous = current


# Defect 8: THD is measured as a peak ratio instead of a harmonic-energy ratio.


def test_thd_uses_harmonic_energy_not_peak_ratio():
    sample_rate = 8000
    freq = 50.0
    count = 1600
    fundamental = [
        math.sin(2.0 * math.pi * freq * index / sample_rate)
        for index in range(count)
    ]
    distorted = [
        fundamental[index]
        + 0.1 * math.sin(2.0 * math.pi * 3.0 * freq * index / sample_rate)
        for index in range(count)
    ]

    assert measure_thd(fundamental, freq, sample_rate) == pytest.approx(
        0.0, abs=1e-6
    )
    thd = measure_thd(distorted, freq, sample_rate)
    assert thd == pytest.approx(0.1, abs=1e-3)
    assert 0.0 <= thd <= 1.0


# Defect 9: the batch shares one hysteresis state, so tracks bleed together.


def test_batch_does_not_share_hysteresis_state_between_tracks():
    track = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    tracks = {"left": list(track), "right": list(track)}
    config = {"drive": 1.0, "mix": 1.0, "width": 0.2}

    output = saturate_batch(tracks, config)
    assert output["left"] == pytest.approx(output["right"], abs=1e-9)


# Defect 10: ``cutoff <= 0`` is not validated, which divides by zero.


def test_dc_block_rejects_non_positive_cutoff():
    with pytest.raises(TapeError):
        dc_block([0.1, 0.2], 0.0)
    with pytest.raises(TapeError):
        dc_block([0.1, 0.2], -0.01)


# Contract checks: five extra regressions that must stay green after a fix.


def test_mix_zero_returns_input():
    samples = [0.0, 0.2, -0.7, 0.999]
    assert saturate(samples, 3.0, 0.0) == pytest.approx(samples, abs=1e-9)


def test_thd_non_decreasing_with_drive():
    sample_rate = 8000
    freq = 100.0
    count = 800
    tone = [
        0.5 * math.sin(2.0 * math.pi * freq * index / sample_rate)
        for index in range(count)
    ]
    thds = [
        measure_thd(saturate(tone, drive, 1.0), freq, sample_rate)
        for drive in (0.5, 1.0, 2.0, 4.0)
    ]
    for lower, higher in zip(thds, thds[1:]):
        assert higher >= lower - 1e-6


def test_empty_input_returns_empty():
    assert saturate([], 1.0, 0.5) == []
    assert dc_block([], 0.01) == []
    assert tape_wobble([], 0.5, 2.0, 8000) == []


def test_drive_zero_raises_tape_error():
    with pytest.raises(TapeError):
        soft_clip(0.3, 0.0)
    with pytest.raises(TapeError):
        saturate([0.3], 0.0, 0.5)


def test_batch_preserves_order_on_success():
    tracks = {
        "voice": [0.1, 0.2, 0.3],
        "guitar": [0.0, 0.1, 0.0],
        "drums": [0.5, -0.5, 0.25],
    }
    output = saturate_batch(tracks, {"drive": 1.5, "mix": 0.5})
    assert list(output) == list(tracks)
    assert all(len(values) == 3 for values in output.values())
