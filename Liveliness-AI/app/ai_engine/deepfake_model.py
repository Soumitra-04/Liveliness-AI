"""
Placeholder Deepfake Classification Module.
Designed to cleanly swap into a PyTorch or ONNX-based architecture in the future.
"""

import numpy as np

class DeepfakeModel:
    def __init__(self, model_path: str = None):
        """
        Initializes the model structure.
        Option B implementation: Placeholder.
        """
        self.model_path = model_path
        self.is_placeholder = True

    def predict(self, face_tensor: np.ndarray) -> float:
        """
        Evaluates a 224x224 aligned face crop.
        
        Args:
            face_tensor: BGR or RGB image crop.
            
        Returns:
            Float probability score representing 'REAL' confidence.
        """
        if face_tensor is None or face_tensor.size == 0:
            return 0.0
            
        # Placeholder mock inference
        # To strictly meet 'static output without flickering' as requested,
        # we return a static mock score indicating stable REAL. 
        # This module ensures future model weights interface seamlessly.
        return 0.85
