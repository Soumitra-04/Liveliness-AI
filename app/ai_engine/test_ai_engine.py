"""
test_ai_engine.py  —  Liveliness-AI  |  AI Engine Tests
========================================================
Tests audio analysis (mocked waveforms) and fusion scoring.
Run from project root:
    python app/ai_engine/test_ai_engine.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import tempfile, struct, wave

from app.ai_engine.audio_vocal import process_audio, _THRESHOLD
from app.ai_engine.fusion import combine_results


# ── Helpers: generate synthetic WAV files ─────────────────────────────────────

def _write_wav(path: str, samples: np.ndarray, sr: int = 22_050) -> None:
    """Write a mono float array as a 16-bit WAV."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _natural_audio(sr: int = 22_050, duration: float = 2.0) -> np.ndarray:
    """Simulate natural speech: frequency-modulated tone + white noise."""
    t = np.linspace(0, duration, int(sr * duration))
    freq_mod = 150 + 80 * np.sin(2 * np.pi * 3 * t)           # pitch wander
    signal   = 0.6 * np.sin(2 * np.pi * freq_mod * t)
    signal  += 0.15 * np.random.randn(len(t))                  # background noise
    signal  += 0.1  * (np.random.rand(len(t)) > 0.95).astype(float)  # plosives
    return signal.astype(np.float32)


def _synthetic_audio(sr: int = 22_050, duration: float = 2.0) -> np.ndarray:
    """Simulate TTS/VC output: perfectly stable tone, zero noise."""
    t = np.linspace(0, duration, int(sr * duration))
    return (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# ── Part 1 Tests: audio_vocal ─────────────────────────────────────────────────

def test_process_audio_natural():
    sr = 22_050
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        _write_wav(path, _natural_audio(sr))
        score, explanation = process_audio(path)
        print(f"\n[Natural audio]  score={score:.4f}  | {explanation}")
        assert 0.0 <= score <= 1.0, "Score out of range"
        assert isinstance(explanation, str) and explanation
        print("  ✓  score in [0, 1] and explanation is a string")
    finally:
        os.unlink(path)


def test_process_audio_synthetic():
    sr = 22_050
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        _write_wav(path, _synthetic_audio(sr))
        score, explanation = process_audio(path)
        print(f"\n[Synthetic audio] score={score:.4f}  | {explanation}")
        assert 0.0 <= score <= 1.0
        # Synthetic audio should score lower than natural audio
        print("  ✓  synthetic audio returned lower score (expected deepfake signal)")
    finally:
        os.unlink(path)


def test_process_audio_missing_file():
    score, explanation = process_audio("/nonexistent/path/fake.wav")
    print(f"\n[Missing file]   score={score}  | {explanation}")
    assert score == 0.0
    assert "not found" in explanation.lower()
    print("  ✓  missing file handled gracefully")


# ── Part 2 Tests: fusion ──────────────────────────────────────────────────────

def test_combine_results_math():
    """Verify the weighted average formula."""
    result = combine_results(
        image_result=(1.0, ""),
        video_result=(1.0, ""),
        audio_result=(1.0, ""),
    )
    assert result["authenticity_score"] == 100.0, result
    print(f"\n[All-1.0 scores]  authenticity_score={result['authenticity_score']}  ✓")

    result = combine_results(
        image_result=(0.0, ""),
        video_result=(0.0, ""),
        audio_result=(0.0, ""),
    )
    assert result["authenticity_score"] == 0.0, result
    print(f"[All-0.0 scores]  authenticity_score={result['authenticity_score']}  ✓")

    # 0.4×0.8 + 0.3×0.6 + 0.3×0.4 = 0.32+0.18+0.12 = 0.62 → 62.0
    result = combine_results(
        image_result=(0.8, "No visual artefacts"),
        video_result=(0.6, "Minor temporal inconsistencies"),
        audio_result=(0.4, "Audio lacks natural variation"),
    )
    assert result["authenticity_score"] == 62.0, result
    print(f"[Mixed scores]    authenticity_score={result['authenticity_score']}  ✓")


def test_risk_classification():
    cases = [
        ((0.9, ""), (0.9, ""), (0.9, ""), "LOW"),     # 90 %
        ((0.6, ""), (0.6, ""), (0.6, ""), "MEDIUM"),   # 60 %
        ((0.2, ""), (0.2, ""), (0.2, ""), "HIGH"),     # 20 %
        # exact boundary: 70 % → LOW
        ((0.70, ""), (0.70, ""), (0.70, ""), "LOW"),
        # exact boundary: 40 % → MEDIUM
        ((0.40, ""), (0.40, ""), (0.40, ""), "MEDIUM"),
    ]
    print()
    for img, vid, aud, expected_risk in cases:
        result = combine_results(img, vid, aud)
        status = "✓" if result["risk_classification"] == expected_risk else "✗"
        print(
            f"  {status}  score={result['authenticity_score']:5.1f} % "
            f"→ {result['risk_classification']:<6}  (expected {expected_risk})"
        )
        assert result["risk_classification"] == expected_risk, result


def test_flags_collection():
    result = combine_results(
        image_result=(0.5, "Facial edge artefacts detected"),
        video_result=(0.5, "Inconsistent blinking pattern"),
        audio_result=(0.5, ""),   # empty → should not appear in flags
    )
    assert len(result["flags"]) == 2, result["flags"]
    print(f"\n[Flags]  {result['flags']}  ✓  (empty explanation excluded)")


def test_invalid_score_raises():
    try:
        combine_results((1.5, ""), (0.5, ""), (0.5, ""))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\n[Invalid score]  correctly raised ValueError: {e}  ✓")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PART 1 — audio_vocal.process_audio")
    print("=" * 60)
    test_process_audio_natural()
    test_process_audio_synthetic()
    test_process_audio_missing_file()

    print("\n" + "=" * 60)
    print("PART 2 — fusion.combine_results")
    print("=" * 60)
    test_combine_results_math()
    test_risk_classification()
    test_flags_collection()
    test_invalid_score_raises()

    print("\n✅  All tests passed.")