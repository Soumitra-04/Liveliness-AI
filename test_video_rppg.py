"""
Liveliness-AI — Standalone Test for Video Analysis (rPPG & FaceMesh)
====================================================================
Verifies OpenCV frame extraction and MediaPipe anomaly detection.
"""

import os
import sys
import cv2
import numpy as np

# Ensure Python can find the 'app' module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.ai_engine.video_rppg import process_video

def create_dummy_video(filename: str = "temp_dummy.mp4"):
    """Generates a 1-second video of random noise (no face)."""
    height, width = 480, 640
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for _ in range(fps): # 30 frames
        # Generate random static noise
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)
        
    out.release()
    return filename

def run_tests():
    print("\n" + "="*60)
    print("🎥 Starting Video Analysis (FaceMesh) Engine Test")
    print("="*60)

    # ---------------------------------------------------------
    # TEST 1: Missing File Handling
    # ---------------------------------------------------------
    print("\n▶ TEST 1: Testing invalid file path...")
    score, explanation = process_video("non_existent_video.mp4")
    print(f"   Score: {score}")
    print(f"   Explanation: {explanation}")
    if score == 0.5 and "Could not open video file" in explanation:
        print("   ✅ PASS: Invalid files handled correctly.")
    else:
        print("   ❌ FAIL: Invalid file handling broke.")

    # ---------------------------------------------------------
    # TEST 2: Valid Video, No Face
    # ---------------------------------------------------------
    print("\n▶ TEST 2: Testing valid video but NO FACE present...")
    dummy_path = create_dummy_video("temp_no_face.mp4")
    
    try:
        score, explanation = process_video(dummy_path)
        print(f"   Score: {score}")
        print(f"   Explanation: {explanation}")
        if "No face detected" in explanation or "too few frames" in explanation:
            print("   ✅ PASS: FaceMesh correctly rejected the static noise video.")
        else:
            print("   ❌ FAIL: Expected a 'no face' warning.")
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path) # Clean up

    # ---------------------------------------------------------
    # TEST 3: Real Video (Requires manual input)
    # ---------------------------------------------------------
    print("\n▶ TEST 3: Testing REAL face video...")
    real_video = "test_face.mp4" 
    
    if os.path.exists(real_video):
        print(f"   Found '{real_video}'. Analyzing landmark math (this takes a moment)...")
        score, explanation = process_video(real_video)
        print(f"\n   🎯 FINAL SCORE: {score}")
        print(f"   📝 EXPLANATION: {explanation}")
        print("   ✅ PASS: Real video processed successfully!")
    else:
        print(f"   ⚠️ SKIP: '{real_video}' not found.")
        print(f"      -> To run this test, record a short webcam video of your face,")
        print(f"         name it '{real_video}', place it in this folder, and run again.")

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    # Note: process_video is synchronous, so we don't need asyncio here
    run_tests()