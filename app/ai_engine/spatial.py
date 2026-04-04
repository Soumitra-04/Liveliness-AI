import cv2
import numpy as np

def spatial_inconsistency_score(file_path: str) -> float:
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Measure irregularity
    score = np.std(edges) / 255

    return min(score, 1.0)