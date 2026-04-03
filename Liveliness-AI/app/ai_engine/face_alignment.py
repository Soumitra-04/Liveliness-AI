"""
Face alignment module using Similarity Transform.
Aligns facial landmarks (5-points) to standard reference coordinates to generate
a stabilized 224x224 crop for downstream deepfake models.
"""

import cv2
import numpy as np

# Standard referenced 5-point facial landmarks for a 112x112 box,
# scaled up by 2.0x for a 224x224 input requirement.
REFERENCE_FACIAL_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

REFERENCE_FACIAL_POINTS_224 = REFERENCE_FACIAL_POINTS * 2.0

def align_face(img: np.ndarray, landmarks: np.ndarray, output_size: tuple = (224, 224)) -> np.ndarray:
    """
    Aligns a face using an affine warp similarity transform to eliminate jitter.
    
    Args:
        img: Original BGR frame.
        landmarks: (5, 2) numpy array of facial keypoints (Left Eye, Right Eye, Nose, Left Mouth, Right Mouth).
        output_size: Target dimension tuple (width, height).
        
    Returns:
        Aligned and cropped BGR image of the specified output_size.
    """
    # Estimate the similarity transformation matrix using LMEDS (Least Median of Squares)
    # which is robust against minor landmark variance.
    tform, _ = cv2.estimateAffinePartial2D(landmarks, REFERENCE_FACIAL_POINTS_224, method=cv2.LMEDS)
    
    if tform is None:
        # Fallback to standard warp dimension if calculation fails severely
        tform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
    # Perform the affine warp
    aligned_face = cv2.warpAffine(img, tform, output_size, borderValue=0.0)
    
    return aligned_face
