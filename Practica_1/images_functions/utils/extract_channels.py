import cv2
import numpy as np
import os

def decompose_image_channels(image_path):
    """
    Read an image from a given path and return both its RGB and grayscale versions,
    along with the separated R, G, and B channels.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: {
            'img_rgb': RGB image,
            'img_gray': grayscale image,
            'R': red channel,
            'G': green channel,
            'B': blue channel
        }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    # Leer imagen en RGB y en escala de grises (sin convertir)
    img_rgb = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img_rgb is None or img_gray is None:
        raise ValueError(f"Error reading image from {image_path}")
    
    # Extraer canales
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    return {
        'img_rgb': img_rgb,
        'img_gray': img_gray,
        'R': R,
        'G': G,
        'B': B
    }


def extract_rgb_channels(image, channels=('R',)):
    if image is None:
        raise ValueError("Input image is None")
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a color image with 3 channels")
    
    channel_map = {'R': 0, 'G': 1, 'B': 2}
    result = np.zeros_like(image)
    for ch in channels:
        ch = ch.upper()
        if ch in channel_map:
            result[:, :, channel_map[ch]] = image[:, :, channel_map[ch]]
        else:
            raise ValueError(f"Invalid channel: {ch}. Valid channels are 'R', 'G', 'B'.")
    return result
