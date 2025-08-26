from typing import Tuple, Optional
import numpy as np
import face_recognition

def extract_face_embedding(image_array: np.ndarray) -> Optional[np.ndarray]:
    """Returns the first face embedding found in the image, or None."""
    # Convert to RGB if needed (face_recognition expects RGB)
    if image_array.shape[-1] == 4:  # RGBA -> RGB
        image_array = image_array[..., :3]
    encodings = face_recognition.face_encodings(image_array)
    if len(encodings) == 0:
        return None
    return encodings[0]

def compute_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return np.linalg.norm(emb1 - emb2)
