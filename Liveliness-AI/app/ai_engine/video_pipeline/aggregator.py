"""
aggregator.py — Aggregate per-frame FAKE-probability scores into a verdict.

Score convention (standardised)
--------------------------------
  All scores represent probability of being FAKE ∈ [0, 1].
    0.0 → certainly real
    1.0 → certainly fake

  Source: inference_engine.py returns probs[:, 1] (class 1 = FAKE).

Decision logic
--------------
  fake_frame_threshold = 0.5
  A frame is FAKE  if score > fake_frame_threshold.

  FAKE verdict when:
    mean_score > 0.6
    OR  (fake_ratio > 0.5  AND  max_score > 0.85)

  REAL otherwise.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

FAKE_FRAME_THRESHOLD = 0.50   # score > this → frame is FAKE
MEAN_FAKE_THRESHOLD  = 0.60   # mean score above this → FAKE
FAKE_RATIO_THRESHOLD = 0.50   # more than half frames fake → trigger
MAX_SCORE_THRESHOLD  = 0.85   # combined with ratio check


def aggregate(
    scores: List[float],
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Combine per-frame FAKE-probability scores into a structured verdict.

    Parameters
    ----------
    scores : List[float]
        FAKE-probability per frame ∈ [0, 1].
    debug : bool
        If True, print diagnostic information to stdout.

    Returns
    -------
    Dict[str, Any]
        {
            "final_verdict":    "REAL" | "FAKE" | "UNCERTAIN",
            "confidence_score": float,   <- mean fake-probability
            "metrics": {
                "frames_analysed": int,
                "frames_fake":     int,
                "frames_real":     int,
                "fake_ratio":      float,
                "mean_score":      float,
                "max_score":       float,
                "min_score":       float,
            }
        }
    """
    if not scores:
        return {
            "final_verdict":    "UNCERTAIN",
            "confidence_score": 0.0,
            "metrics": {
                "frames_analysed": 0,
                "frames_fake":     0,
                "frames_real":     0,
                "fake_ratio":      0.0,
                "mean_score":      0.0,
                "max_score":       0.0,
                "min_score":       0.0,
            },
        }

    n           = len(scores)
    mean_score  = sum(scores) / n
    max_score   = max(scores)
    min_score   = min(scores)
    frames_fake = sum(1 for s in scores if s > FAKE_FRAME_THRESHOLD)
    frames_real = n - frames_fake
    fake_ratio  = frames_fake / n

    # ── Debug output ──────────────────────────────────────────────────────────
    if debug:
        print(f"\n[aggregator] DEBUG ─────────────────────────────")
        print(f"  frames_analysed : {n}")
        print(f"  mean_score      : {mean_score:.4f}  (threshold > {MEAN_FAKE_THRESHOLD} → FAKE)")
        print(f"  fake_ratio      : {fake_ratio:.4f}  (threshold > {FAKE_RATIO_THRESHOLD})")
        print(f"  max_score       : {max_score:.4f}  (for ratio check > {MAX_SCORE_THRESHOLD})")
        print(f"  frames_fake     : {frames_fake}")
        sample = scores[:10]
        print(f"  sample scores   : {[round(s,3) for s in sample]}")
        print(f"─────────────────────────────────────────────────\n",
              file=sys.stdout, flush=True)

    # ── Decision ──────────────────────────────────────────────────────────────
    if mean_score > MEAN_FAKE_THRESHOLD or \
       (fake_ratio > FAKE_RATIO_THRESHOLD and max_score > MAX_SCORE_THRESHOLD):
        verdict = "FAKE"
    else:
        verdict = "REAL"

    return {
        "final_verdict":    verdict,
        "confidence_score": round(mean_score, 4),
        "metrics": {
            "frames_analysed": n,
            "frames_fake":     frames_fake,
            "frames_real":     frames_real,
            "fake_ratio":      round(fake_ratio, 4),
            "mean_score":      round(mean_score, 4),
            "max_score":       round(max_score, 4),
            "min_score":       round(min_score, 4),
        },
    }
