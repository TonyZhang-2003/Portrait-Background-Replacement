import numpy as np
from PIL import Image
import cv2

def load_image(image_path):
    """Load image from path or PIL Image"""
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path
    return image


def refine_mask(mask):
    """
    Edge-aware mask refinement (hair-friendly)
    - smooths edges
    - preserves fine details
    - NO ximgproc required
    """
    mask = mask.astype(np.uint8)

    # Edge-preserving smoothing
    mask = cv2.bilateralFilter(mask, d=7, sigmaColor=60, sigmaSpace=60)

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Feather edges (important for hair)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask


def apply_mask(image, mask, background):
    """
    Alpha blending with soft edges
    """
    img = np.array(image).astype(np.float32)
    bg = np.array(background.resize(image.size)).astype(np.float32)

    alpha = mask.astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=2)

    result = img * alpha + bg * (1 - alpha)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)
