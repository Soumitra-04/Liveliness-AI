"""
audio_vocal.py  —  Liveliness-AI  |  Audio Analysis Module
===========================================================
Extracts low-level acoustic features from an audio file and
produces an authenticity score (0 → likely fake, 1 → likely real).

Features extracted
------------------
  • MFCC              — timbral texture; synthetic voices are suspiciously
                        uniform across frames
  • Spectral centroid  — centre of mass of the spectrum; TTS/VC voices
                        cluster in a narrow band
  • Zero-crossing rate — proxy for signal noisiness; natural speech is
                        messy, cloned speech is clean

Primary backend : librosa   (install: pip install librosa)
Fallback backend: scipy + numpy   (used when librosa is not available)
"""

from __future__ import annotations

import os
import warnings
from typing import Tuple

import numpy as np

# ── Backend selection ─────────────────────────────────────────────────────────

try:
    import librosa  # type: ignore
    _BACKEND = "librosa"
except ImportError:  # pragma: no cover
    librosa = None   # type: ignore
    _BACKEND = "scipy"

# ── Public API ────────────────────────────────────────────────────────────────

def process_audio(file_path: str) -> Tuple[float, str]:
    """
    Analyse an audio file and return a deepfake authenticity score.

    Parameters
    ----------
    file_path : str
        Path to the audio file (.wav or .mp3 recommended).

    Returns
    -------
    score : float
        Value in [0.0, 1.0].
        High score (→ 1) = sounds natural / authentic.
        Low  score (→ 0) = sounds synthetic / manipulated.

    explanation : str
        Human-readable summary of what triggered (or didn't trigger)
        the suspicion flags.

    Example
    -------
    >>> score, explanation = process_audio("sample.wav")
    >>> score
    0.4
    >>> explanation
    "Audio lacks natural variation"
    """
    if not os.path.isfile(file_path):
        return 0.0, f"File not found: {file_path}"

    try:
        y, sr = _load_audio(file_path)
    except Exception as exc:
        return 0.0, f"Could not load audio file: {exc}"

    if len(y) == 0:
        return 0.0, "Audio file is empty or unreadable."

    # ── Feature extraction ────────────────────────────────────────────────────
    mfcc_matrix      = _extract_mfcc(y, sr)          # shape (n_mfcc, frames)
    spectral_centroid = _extract_spectral_centroid(y, sr)  # 1-D array
    zcr               = _extract_zcr(y)                    # 1-D array

    # ── Heuristic checks → suspicion flags ───────────────────────────────────
    flags: list[str] = []
    penalty: float   = 0.0

    # 1. MFCC temporal variance — synthetic voices have unnaturally flat
    #    MFCC trajectories across time.
    mfcc_var = float(np.mean(np.var(mfcc_matrix, axis=1)))
    if mfcc_var < _THRESHOLD["mfcc_var_low"]:
        flags.append("MFCC variance too low — overly smooth / robotic timbre")
        penalty += 0.30

    # 2. Spectral centroid variation — a voice with almost constant
    #    brightness is a strong TTS/VC indicator.
    sc_std = float(np.std(spectral_centroid))
    if sc_std < _THRESHOLD["sc_std_low"]:
        flags.append("Spectral centroid nearly constant — lack of natural pitch movement")
        penalty += 0.25

    # 3. Zero-crossing rate std — real speech has bursts of frication /
    #    silence that raise ZCR variance; cloned audio is suspiciously steady.
    zcr_std = float(np.std(zcr))
    if zcr_std < _THRESHOLD["zcr_std_low"]:
        flags.append("Zero-crossing rate too uniform — likely synthetic signal")
        penalty += 0.20

    # 4. MFCC delta flatness — first-order MFCC deltas measure how quickly
    #    the timbre changes; near-zero deltas = robotically smooth transitions.
    mfcc_delta_mean = float(np.mean(np.abs(np.diff(mfcc_matrix, axis=1))))
    if mfcc_delta_mean < _THRESHOLD["mfcc_delta_low"]:
        flags.append("MFCC transitions too smooth — audio lacks natural variation")
        penalty += 0.25

    # ── Score assembly ────────────────────────────────────────────────────────
    raw_score = max(0.0, 1.0 - penalty)
    score     = round(float(np.clip(raw_score, 0.0, 1.0)), 4)

    if not flags:
        explanation = "Audio features appear natural — no synthetic indicators detected."
    elif len(flags) == 1:
        explanation = flags[0]
    else:
        explanation = "Multiple synthetic indicators detected: " + "; ".join(flags)

    return score, explanation


# ── Internal helpers ──────────────────────────────────────────────────────────

# Empirically tuned thresholds.
# These are conservative defaults; tune per dataset once labelled data exists.
_THRESHOLD: dict[str, float] = {
    "mfcc_var_low":    10.0,   # below this → MFCC too flat
    "sc_std_low":     200.0,   # Hz; below this → spectral centroid too stable
    "zcr_std_low":      0.02,  # below this → ZCR too uniform
    "mfcc_delta_low":   0.5,   # below this → transitions too smooth
}


def _load_audio(file_path: str, target_sr: int = 22_050) -> Tuple[np.ndarray, int]:
    """Load audio to a mono float32 waveform at *target_sr* Hz."""
    if _BACKEND == "librosa":
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return y, sr

    # ── scipy fallback (WAV only) ─────────────────────────────────────────
    from scipy.io import wavfile  # type: ignore
    from scipy.signal import resample  # type: ignore

    sr_orig, data = wavfile.read(file_path)

    # Convert to float in [-1, 1]
    if data.dtype.kind == "i":
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    elif data.dtype.kind == "u":
        data = (data.astype(np.float32) - 128) / 128.0
    else:
        data = data.astype(np.float32)

    # Mix down to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if necessary
    if sr_orig != target_sr:
        num_samples = int(len(data) * target_sr / sr_orig)
        data = resample(data, num_samples).astype(np.float32)

    return data, target_sr


def _extract_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """Return MFCC matrix of shape (n_mfcc, n_frames)."""
    if _BACKEND == "librosa":
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # ── Manual MFCC via scipy ─────────────────────────────────────────────
    from scipy.fft import dct  # type: ignore

    frame_len  = int(sr * 0.025)   # 25 ms
    hop_len    = int(sr * 0.010)   # 10 ms
    n_fft      = 512
    n_mels     = 26

    frames = _frame_signal(y, frame_len, hop_len)
    window = np.hanning(frame_len)
    spec   = np.abs(np.fft.rfft(frames * window[:frames.shape[1]], n=n_fft)) ** 2

    mel_fb  = _mel_filterbank(sr, n_fft, n_mels)
    mel_spec = np.dot(spec, mel_fb.T)
    log_mel  = np.log(mel_spec + 1e-9)

    mfcc = dct(log_mel, type=2, norm="ortho", axis=1)[:, :n_mfcc]
    return mfcc.T  # (n_mfcc, n_frames)


def _extract_spectral_centroid(y: np.ndarray, sr: int) -> np.ndarray:
    """Return per-frame spectral centroid in Hz."""
    if _BACKEND == "librosa":
        return librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    frame_len = int(sr * 0.025)
    hop_len   = int(sr * 0.010)
    n_fft     = 512

    frames   = _frame_signal(y, frame_len, hop_len)
    window   = np.hanning(frame_len)
    mag_spec = np.abs(np.fft.rfft(frames * window[:frames.shape[1]], n=n_fft))
    freqs    = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    weights    = mag_spec * freqs[np.newaxis, :]
    total_mag  = mag_spec.sum(axis=1) + 1e-9
    centroid   = weights.sum(axis=1) / total_mag
    return centroid


def _extract_zcr(y: np.ndarray, frame_len: int = 512, hop_len: int = 256) -> np.ndarray:
    """Return per-frame zero-crossing rate."""
    if _BACKEND == "librosa":
        return librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]

    frames = _frame_signal(y, frame_len, hop_len)
    signs  = np.sign(frames)
    # zero-crossings = sign changes between consecutive samples
    zcr    = (np.diff(signs, axis=1) != 0).sum(axis=1) / (frame_len - 1)
    return zcr.astype(np.float32)


# ── Very-low-level signal utilities (scipy fallback only) ────────────────────

def _frame_signal(y: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Slice 1-D signal into overlapping frames; returns (n_frames, frame_len)."""
    n_frames = 1 + (len(y) - frame_len) // hop_len
    idx      = (
        np.arange(frame_len)[np.newaxis, :]
        + np.arange(n_frames)[:, np.newaxis] * hop_len
    )
    return y[idx]


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Triangular mel filterbank; returns (n_mels, n_fft//2 + 1)."""
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_mel  = hz_to_mel(0.0)
    high_mel = hz_to_mel(sr / 2.0)
    mel_pts  = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts   = np.array([mel_to_hz(m) for m in mel_pts])
    bin_pts  = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    n_bins = n_fft // 2 + 1
    fb     = np.zeros((n_mels, n_bins))
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(lo, ctr):
            fb[m - 1, k] = (k - lo) / (ctr - lo + 1e-9)
        for k in range(ctr, hi):
            fb[m - 1, k] = (hi - k) / (hi - ctr + 1e-9)
    return fb