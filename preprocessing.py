"""
Unified Fingerprint Preprocessing and Augmentation Module
=========================================================
Implements identical preprocessing for training, evaluation, and inference:
1. Image loading (filepath, PIL Image, bytes, or numpy array)
2. Grayscale conversion
3. CLAHE (Contrast Limited Adaptive Histogram Equalization) for ridge enhancement
4. Resizing to target dimensions (224x224 or 256x256)
5. Architecture-specific normalization (VGG16, ResNet50, AlexNet, LeNet)
6. Domain-safe biometric data augmentation applied ONLY during training
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input

TARGET_IMAGE_SIZE_VGG16 = (224, 224)
TARGET_IMAGE_SIZE = (224, 224)
TARGET_IMAGE_SIZE_RESNET = (256, 256)
TARGET_IMAGE_SIZE_ALEXNET = (224, 224)
TARGET_IMAGE_SIZE_LENET = (64, 64)

CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID_SIZE = (8, 8)


def load_image_to_pil(image_source) -> Image.Image:
    """
    Load an image from filepath, PIL Image, bytes, or numpy array into a PIL Image.
    """
    if isinstance(image_source, Image.Image):
        return image_source.copy()
    elif isinstance(image_source, str):
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"Image not found at path: {image_source}")
        return Image.open(image_source)
    elif isinstance(image_source, (bytes, bytearray)):
        import io
        return Image.open(io.BytesIO(image_source))
    elif isinstance(image_source, np.ndarray):
        return Image.fromarray(image_source)
    else:
        raise TypeError(f"Unsupported image source type: {type(image_source)}")


def apply_clahe_enhancement(
    gray_arr: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_size: tuple = CLAHE_TILE_GRID_SIZE
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to accentuate ridge and valley contours in fingerprint impressions.
    """
    if gray_arr.dtype != np.uint8:
        gray_arr = np.clip(gray_arr, 0, 255).astype(np.uint8)
        
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(gray_arr)
    return enhanced


def preprocess_fingerprint(
    image_source,
    target_size: tuple = TARGET_IMAGE_SIZE_VGG16,
    apply_clahe: bool = True,
    model_type: str = 'vgg16'
) -> np.ndarray:
    """
    Standard preprocessing pipeline used identically during training and inference.
    
    Args:
        image_source: Filepath, PIL Image, or numpy array.
        target_size: Output image dimensions.
        apply_clahe: Whether to apply CLAHE ridge enhancement.
        model_type: One of 'vgg16', 'resnet', 'alexnet', 'lenet', 'raw'.

    Returns:
        Preprocessed numpy array of shape (target_size[0], target_size[1], 3) or (H, W, 1).
    """
    # 1. Load image
    pil_img = load_image_to_pil(image_source)
    
    # 2. Convert to Grayscale
    gray_img = pil_img.convert('L')
    gray_np = np.array(gray_img, dtype=np.uint8)

    # 3. Contrast enhancement via CLAHE
    if apply_clahe:
        enhanced_np = apply_clahe_enhancement(gray_np, clip_limit=CLAHE_CLIP_LIMIT, tile_size=CLAHE_TILE_GRID_SIZE)
    else:
        enhanced_np = gray_np

    # 4. Resize to target dimensions
    enhanced_pil = Image.fromarray(enhanced_np)
    resized_pil = enhanced_pil.resize(target_size, Image.Resampling.BICUBIC)
    
    # 5. Expand to 3-channel RGB for CNN backbones
    rgb_pil = resized_pil.convert('RGB')
    rgb_arr = np.array(rgb_pil, dtype=np.float32)

    # 6. Apply architecture-specific normalization
    m_type = model_type.lower()
    if 'vgg' in m_type:
        processed = vgg16_preprocess_input(rgb_arr)
    elif 'resnet' in m_type:
        processed = resnet50_preprocess_input(rgb_arr)
    elif m_type in ['alexnet', 'lenet']:
        processed = rgb_arr / 255.0
    else:
        processed = rgb_arr / 255.0

    return processed


def augment_fingerprint(pil_img: Image.Image, rng: np.random.RandomState = None) -> Image.Image:
    """
    Apply domain-safe fingerprint augmentation:
    - Small rotations (-10 deg to +10 deg)
    - Small translations (+- 4 pixels)
    - Slight zoom / crop (0.95x - 1.05x)
    - Mild contrast and brightness variations (+- 10%)
    Preserves biometric ridge-valley integrity.
    Applied ONLY on the training split!
    """
    if rng is None:
        rng = np.random.RandomState()

    img = pil_img.convert('L')
    
    # 1. Small random rotation (-8 to +8 degrees)
    angle = rng.uniform(-8.0, 8.0)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
    
    # 2. Small random translation
    w, h = img.size
    dx = rng.randint(-4, 5)
    dy = rng.randint(-4, 5)
    img = img.transform((w, h), Image.Transform.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=255)

    # 3. Random contrast adjustment (0.90 to 1.10)
    enh_contrast = ImageEnhance.Contrast(img)
    img = enh_contrast.enhance(rng.uniform(0.90, 1.10))

    # 4. Random brightness adjustment (0.92 to 1.08)
    enh_bright = ImageEnhance.Brightness(img)
    img = enh_bright.enhance(rng.uniform(0.92, 1.08))

    return img
