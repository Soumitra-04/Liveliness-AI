import sys
import os
import cv2
import time
import threading
import tempfile
from collections import deque
import numpy as np

# Ensure Python can find your AI engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.ai_engine.video_inference import process_video

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
FRAME_BUFFER_SIZE = 180       # ~6 seconds of video for the FFT to analyze
FPS_TARGET = 30               # Target FPS for the temp video file
NO_FACE_RESET = 15            # Clear buffer if face is lost for this many frames

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE (Thread-safe-ish for GUI)
# ═════════════════════════════════════════════════════════════════════════════
frames_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
is_processing = False
last_process_time = 0

# UI State
current_score = None
current_verdict = "CALIBRATING SENSORS..."
current_color = (0, 255, 255) # Yellow
current_explanation = ""
face_bbox = None

# OpenCV fast face detector (strictly for drawing the UI box)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# ═════════════════════════════════════════════════════════════════════════════
# BACKGROUND AI THREAD
# ═════════════════════════════════════════════════════════════════════════════
def run_ai_analysis(snapshot_frames):
    """Runs the heavy rPPG math without freezing the webcam."""
    global current_score, current_verdict, current_color, current_explanation, is_processing

    # 1. Write the snapshot to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        h, w = snapshot_frames[0].shape[:2]
        # XVID is the most stable codec across Windows/Mac/Linux for temp OpenCV files
        out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*"XVID"), FPS_TARGET, (w, h))
        for f in snapshot_frames:
            out.write(f)
        out.release()

        # 2. Run the actual AI Engine
        score, explanation = process_video(temp_path)

        # 3. Update the UI Global Variables
        current_score = score
        current_explanation = explanation

        if score >= 0.70:
            current_verdict = f"REAL HUMAN ({score*100:.1f}%)"
            current_color = (0, 255, 0) # Green
        elif score >= 0.40:
            current_verdict = f"UNCERTAIN ({score*100:.1f}%)"
            current_color = (0, 165, 255) # Orange
        else:
            current_verdict = f"DEEPFAKE DETECTED ({score*100:.1f}%)"
            current_color = (0, 0, 255) # Red

    except Exception as e:
        print(f"❌ AI Thread Error: {e}")
        current_verdict = "ANALYSIS ERROR"
        current_color = (0, 0, 255)
    finally:
        # 4. Clean up and unlock
        if os.path.exists(temp_path):
            os.remove(temp_path)
        is_processing = False

# ═════════════════════════════════════════════════════════════════════════════
# MAIN HUD & WEBCAM LOOP
# ═════════════════════════════════════════════════════════════════════════════
def draw_hud(frame, fps_text, buffer_fill_ratio):
    """Draws a futuristic, hackathon-ready UI over the frame."""
    h, w = frame.shape[:2]

    # --- Top Banner ---
    cv2.rectangle(frame, (0, 0), (w, 60), (15, 15, 15), -1)
    cv2.putText(frame, "LIVENESS-AI MULTIMODAL TRUST ENGINE", (15, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps_text}", (w - 120, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- Verdict & Score ---
    cv2.putText(frame, current_verdict, (15, 100), 
                cv2.FONT_HERSHEY_DUPLEX, 1.0, current_color, 2)

    # --- Buffer Progress Bar ---
    bar_width = 300
    bar_height = 15
    bar_x = 15
    bar_y = 120
    
    # Background of bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    # Fill of bar
    fill_width = int(bar_width * buffer_fill_ratio)
    if is_processing:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 0, 255), -1)
        cv2.putText(frame, "ANALYZING BIOMETRICS...", (bar_x + bar_width + 15, bar_y + 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    else:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 255), -1)

    # --- Explanation Text (Bottom) ---
    if current_explanation:
        cv2.rectangle(frame, (0, h - 40), (w, h), (15, 15, 15), -1)
        # Truncate text if it's too long for the screen
        disp_text = current_explanation[:100] + "..." if len(current_explanation) > 100 else current_explanation
        cv2.putText(frame, disp_text, (15, h - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # --- Face Bounding Box ---
    global face_bbox
    if face_bbox is not None:
        fx, fy, fw, fh = face_bbox
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), current_color, 2)
        # Draw corner brackets for sci-fi feel
        length = 20
        cv2.line(frame, (fx, fy), (fx + length, fy), current_color, 4)
        cv2.line(frame, (fx, fy), (fx, fy + length), current_color, 4)
        cv2.line(frame, (fx + fw, fy), (fx + fw - length, fy), current_color, 4)
        cv2.line(frame, (fx + fw, fy), (fx + fw, fy + length), current_color, 4)


def main():
    global is_processing, current_verdict, current_color, face_bbox

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("🚀 Liveness-AI Engine Started. Press 'q' to quit.")

    no_face_counter = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        # Fast Face Detection for the UI Box
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))

        if len(faces) > 0:
            # Pick the largest face
            face_bbox = max(faces, key=lambda rect: rect[2] * rect[3])
            no_face_counter = 0
            frames_buffer.append(frame)
        else:
            face_bbox = None
            no_face_counter += 1

        # Reset if face is lost for too long
        if no_face_counter > NO_FACE_RESET:
            frames_buffer.clear()
            if not is_processing:
                current_verdict = "NO SUBJECT DETECTED"
                current_color = (100, 100, 100)

        # -------------------------------------------------------------
        # THE MAGIC THREAD TRIGGER
        # -------------------------------------------------------------
        if len(frames_buffer) == FRAME_BUFFER_SIZE and not is_processing:
            is_processing = True
            
            # Take a copy of the frames so the webcam loop can continue immediately
            snapshot = list(frames_buffer)
            
            # Start the AI in the background
            ai_thread = threading.Thread(target=run_ai_analysis, args=(snapshot,))
            ai_thread.daemon = True
            ai_thread.start()
            
            # Clear half the buffer so we continuously re-evaluate every ~2 seconds
            # instead of waiting a full 4 seconds for the next trigger.
            for _ in range(FRAME_BUFFER_SIZE // 2):
                frames_buffer.popleft()

        # Draw UI
        fill_ratio = len(frames_buffer) / FRAME_BUFFER_SIZE
        draw_hud(frame, str(fps), fill_ratio)

        cv2.imshow("Liveness-AI : Real-Time Trust Engine", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()