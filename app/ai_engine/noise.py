import cv2
import numpy as np

def noise_score(file_path: str) -> float:
    img = cv2.imread(file_path, 0)

    # Laplacian detects noise / sharpness
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    score = np.var(laplacian) / 1000
    return min(score, 1.0)