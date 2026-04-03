"""
video_analyzer.py — Orchestrates the full deepfake detection pipeline.

Steps
-----
1. Extract up to 50 frames uniformly at 2 FPS.
2. Detect the largest face in each frame — skip frames with no face.
3. Align face crops to stable 224×224 patches.
4. Run batch inference (16 faces per forward pass) through EfficientNet-B0.
5. Aggregate per-frame scores → final verdict + structured metrics.

Model note
----------
The EfficientNet-B0 backbone is initialised from torchvision for architecture
shape only.  ALL weights are immediately replaced by best_model-v3.pt —
a binary deepfake classifier (0=REAL, 1=FAKE) fine-tuned on deepfake datasets.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.ai_engine.video_pipeline.frame_extractor  import extract_frames
from app.ai_engine.video_pipeline.face_detector    import detect_largest_face
from app.ai_engine.video_pipeline.face_aligner     import align_face
from app.ai_engine.video_pipeline.inference_engine import run_batch_inference
from app.ai_engine.video_pipeline.aggregator       import aggregate


def analyze_video(video_path: str) -> Dict[str, Any]:
    """
    Run end-to-end deepfake detection on a video file.

    Parameters
    ----------
    video_path : str  Path to the saved video file.

    Returns
    -------
    Dict[str, Any]
        Structured result: verdict, confidence, metrics, frame_data.

    Raises
    ------
    ValueError   If no frames could be extracted.
    RuntimeError If no faces were detected across all frames.
    """
    # ── 1. Extract frames ──────────────────────────────────────────────────────
    frames = extract_frames(video_path, target_fps=2.0)
    if not frames:
        raise ValueError("No frames could be extracted from the video.")

    # ── 2–3. Face detection + alignment ───────────────────────────────────────
    valid_faces:  List           = []   # aligned face crops
    face_indices: List[int]      = []   # which frame index each belongs to
    frame_data:   List[Dict]     = []
    n_no_face = 0

    for idx, frame in enumerate(frames):
        box = detect_largest_face(frame)
        if box is None:
            n_no_face += 1
            frame_data.append({
                "frame_index":   idx,
                "face_detected": False,
                "real_score":    None,
                "verdict":       "NO_FACE",
            })
            continue

        face = align_face(frame, box)
        if face is None:
            n_no_face += 1
            frame_data.append({
                "frame_index":   idx,
                "face_detected": False,
                "real_score":    None,
                "verdict":       "ALIGN_FAIL",
            })
            continue

        valid_faces.append(face)
        face_indices.append(idx)
        # Placeholder — will be filled after batch inference
        frame_data.append({
            "frame_index":   idx,
            "face_detected": True,
            "fake_score":    None,
            "verdict":       None,
        })

    if not valid_faces:
        raise RuntimeError(
            f"No faces detected in any of the {len(frames)} extracted frames."
        )

    # ── 4. Batch inference ─────────────────────────────────────────────────────
    # scores = FAKE-probability per face ∈ [0, 1]
    scores: List[float] = run_batch_inference(valid_faces)

    # Back-fill frame_data with fake-scores
    for face_pos, (frame_idx, score) in enumerate(zip(face_indices, scores)):
        # Find the matching frame_data entry (same frame_index, face_detected=True)
        for entry in frame_data:
            if entry["frame_index"] == frame_idx and entry["face_detected"]:
                entry["fake_score"] = round(score, 4)
                entry["verdict"]    = "FAKE" if score > 0.50 else "REAL"
                break

    # ── 5. Aggregate (debug=True prints diagnostics to server log) ─────────────
    result = aggregate(scores, debug=True)
    result["frames_total"]           = len(frames)
    result["frames_skipped_no_face"] = n_no_face
    result["frame_data"]             = frame_data

    return result
