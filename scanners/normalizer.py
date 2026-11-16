"""
Biometric Image Normalizer & Quality Assessment
===============================================
Normalizes raw scanner captures from varied hardware sensor resolutions and bit depths
into the canonical format expected by the VGG16/ResNet and SVM preprocessing pipelines.
"""

import io
import base64
import numpy as np
from PIL import Image, ImageOps
import cv2


MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB safety limit
MIN_DIMENSION = 32                  # Minimum viable pixel dimension for ridge processing


def validate_fingerprint_image(raw_source) -> None:
    """Validates raw image source to reject empty, oversized, or unreadable inputs."""
    if raw_source is None:
        raise ValueError("Image input is empty or None.")
    
    if isinstance(raw_source, (bytes, bytearray)):
        if len(raw_source) == 0:
            raise ValueError("Image byte buffer is empty (0 bytes).")
        if len(raw_source) > MAX_IMAGE_BYTES:
            raise ValueError(f"Image file exceeds maximum allowable size of {MAX_IMAGE_BYTES // (1024*1024)}MB.")
    elif isinstance(raw_source, np.ndarray):
        if raw_source.size == 0:
            raise ValueError("Image array is empty (0 elements).")
    elif isinstance(raw_source, Image.Image):
        w, h = raw_source.size
        if w < MIN_DIMENSION or h < MIN_DIMENSION:
            raise ValueError(f"Image dimensions ({w}x{h}) are too small for biometric analysis. Minimum: {MIN_DIMENSION}x{MIN_DIMENSION}.")


def normalize_fingerprint_image(
    raw_source,
    target_size=(512, 512),
    maintain_aspect_ratio=True,
    auto_contrast=True
) -> tuple[Image.Image, bytes, str, dict]:
    """
    Normalizes a fingerprint capture from a scanner into canonical PIL Image and PNG bytes.

    Args:
        raw_source: PIL Image, numpy array (grayscale or RGB/RGBA), or raw bytes.
        target_size: Desired canonical dimensions (W, H). Default (512, 512).
        maintain_aspect_ratio: If True, pads letterbox edges instead of stretching.
        auto_contrast: If True, balances histogram dynamic range.

    Returns:
        tuple: (pil_image, png_bytes, base64_data_url, quality_metrics)
    """
    # 0. Validation
    validate_fingerprint_image(raw_source)

    # 1. Convert to PIL Image
    if isinstance(raw_source, Image.Image):
        pil_img = raw_source.copy()
    elif isinstance(raw_source, np.ndarray):
        if raw_source.dtype != np.uint8:
            raw_source = np.clip(raw_source, 0, 255).astype(np.uint8)
        if len(raw_source.shape) == 2:
            pil_img = Image.fromarray(raw_source, mode='L')
        elif len(raw_source.shape) == 3:
            if raw_source.shape[2] == 4:
                pil_img = Image.fromarray(raw_source, mode='RGBA').convert('L')
            elif raw_source.shape[2] == 3:
                pil_img = Image.fromarray(raw_source, mode='RGB').convert('L')
            else:
                pil_img = Image.fromarray(raw_source[:, :, 0], mode='L')
        else:
            raise ValueError(f"Invalid numpy array shape for fingerprint: {raw_source.shape}")
    elif isinstance(raw_source, (bytes, bytearray)):
        try:
            pil_img = Image.open(io.BytesIO(raw_source))
        except Exception as e:
            raise ValueError(f"Could not decode image bytes: {str(e)}")
    else:
        raise TypeError(f"Unsupported raw_source type: {type(raw_source)}")

    w, h = pil_img.size
    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        raise ValueError(f"Image dimensions ({w}x{h}) are too small for biometric analysis. Minimum: {MIN_DIMENSION}x{MIN_DIMENSION}.")


    # 2. Convert to Grayscale
    gray_img = pil_img.convert('L')

    # 3. Handle Polarity (Ridge should be darker than platen background for optical sensors)
    np_gray = np.array(gray_img, dtype=np.uint8)
    mean_val = float(np.mean(np_gray))
    # If the image is mostly dark (inverted capacitive / dark-field scanner), invert to standard white-platen
    if mean_val < 80.0:
        gray_img = ImageOps.invert(gray_img)
        np_gray = np.array(gray_img, dtype=np.uint8)

    # 4. Optional Auto-Contrast Normalization
    if auto_contrast:
        gray_img = ImageOps.autocontrast(gray_img, cutoff=1)
        np_gray = np.array(gray_img, dtype=np.uint8)

    # 5. Aspect-Ratio Preserving Resizing / Centering
    orig_w, orig_h = gray_img.size
    target_w, target_h = target_size

    if maintain_aspect_ratio:
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        resized = gray_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Create centered canvas with background matching platen (median border value or 255)
        border_pixels = np.concatenate([
            np_gray[0, :], np_gray[-1, :], np_gray[:, 0], np_gray[:, -1]
        ])
        bg_val = int(np.median(border_pixels)) if len(border_pixels) > 0 else 255

        normalized_pil = Image.new('L', (target_w, target_h), color=bg_val)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        normalized_pil.paste(resized, (paste_x, paste_y))
    else:
        normalized_pil = gray_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # 6. Quality Metrics Assessment (NFIQ-inspired clarity, sharpness, contrast)
    quality_metrics = compute_capture_quality(normalized_pil)

    # 7. Export PNG Bytes and Base64 Data URL
    buf = io.BytesIO()
    normalized_pil.save(buf, format='PNG', optimize=True)
    png_bytes = buf.getvalue()
    b64_str = base64.b64encode(png_bytes).decode('ascii')
    data_url = f"data:image/png;base64,{b64_str}"

    return normalized_pil, png_bytes, data_url, quality_metrics


def compute_capture_quality(pil_img: Image.Image) -> dict:
    """
    Computes quality scores for a captured fingerprint:
    - Laplacian edge sharpness (focus measure)
    - Dynamic range / contrast ratio
    - Foreground ridge area coverage
    - Overall biometric quality index (0 - 100)
    """
    arr = np.array(pil_img.convert('L'), dtype=np.uint8)
    
    # 1. Sharpness via Laplacian variance
    lap = cv2.Laplacian(arr, cv2.CV_64F)
    sharpness_var = float(np.var(lap))
    sharpness_score = min(100.0, (sharpness_var / 800.0) * 100.0)

    # 2. Dynamic Range / Michelson contrast
    p5 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))
    dynamic_range = max(0.0, p95 - p5)
    contrast_score = min(100.0, (dynamic_range / 200.0) * 100.0)

    # 3. Foreground Ridge Coverage (fraction of non-background pixels)
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    ridge_mask = arr < (mean_val - 0.25 * std_val)
    coverage = float(np.mean(ridge_mask)) * 100.0
    coverage_score = min(100.0, coverage * 2.5)

    # 4. Composite Quality Index
    overall_quality = round(0.4 * sharpness_score + 0.35 * contrast_score + 0.25 * coverage_score, 1)
    overall_quality = max(5.0, min(99.0, overall_quality))

    if overall_quality >= 75.0:
        verdict = "Excellent Ridge Clarity"
    elif overall_quality >= 55.0:
        verdict = "Acceptable Diagnostic Quality"
    else:
        verdict = "Low Quality - Re-align Finger"

    return {
        'overall_score': overall_quality,
        'sharpness_score': round(sharpness_score, 1),
        'contrast_score': round(contrast_score, 1),
        'coverage_score': round(coverage_score, 1),
        'dynamic_range': round(dynamic_range, 1),
        'verdict': verdict
    }
