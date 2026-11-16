"""
================================================================================
Fingerprint-Based Blood Group Prediction - Inference Pipeline
================================================================================
Executes identical preprocessing, VGG16 feature extraction, scaling, and SVM
prediction as used during training.
================================================================================
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from PIL import Image

# Output UTF-8 handling for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16

from preprocessing import (
    preprocess_fingerprint,
    TARGET_IMAGE_SIZE
)

MODEL_BUNDLE_PATH = os.path.join('models', 'vgg16_svm_blood_group_model.pkl')
LABEL_MAP_PATH = os.path.join('models', 'label_mapping.json')

# Lazy-loaded global instances
_vgg16_model = None
_model_bundle = None


def get_vgg16_extractor():
    """Load or retrieve the cached VGG16 feature extractor."""
    global _vgg16_model
    if _vgg16_model is None:
        _vgg16_model = VGG16(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        _vgg16_model.trainable = False
    return _vgg16_model


def load_trained_model_bundle(model_path: str = MODEL_BUNDLE_PATH):
    """Load the trained SVM model bundle (classifier, scaler, metadata)."""
    global _model_bundle
    if _model_bundle is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model bundle not found at '{model_path}'. "
                f"Please train the model first by running: python train.py"
            )
        with open(model_path, 'rb') as f:
            _model_bundle = pickle.load(f)
    return _model_bundle


def predict_blood_group(image_input, model_path: str = MODEL_BUNDLE_PATH) -> dict:
    """
    Predict ABO blood group from a fingerprint image.
    
    Args:
        image_input: Filepath, PIL Image, or raw image bytes.
        model_path: Path to the trained model bundle pkl.

    Returns:
        Dictionary containing prediction, confidence, probability breakdown, and metadata.
    """
    # 1. Load model bundle
    bundle = load_trained_model_bundle(model_path)
    svm_model = bundle['svm_model']
    scaler = bundle['scaler']
    label_map = bundle['label_mapping']
    target_classes = bundle.get('target_classes', list(label_map.keys()))

    # Inverted mapping: index -> class name
    inv_label_map = {v: k for k, v in label_map.items()}

    # 2. Consistent Preprocessing (Grayscale -> CLAHE -> Resize -> RGB -> VGG16 preprocess_input)
    processed_arr = preprocess_fingerprint(
        image_input,
        target_size=TARGET_IMAGE_SIZE,
        apply_clahe=True,
        model_type='vgg16'
    )

    # 3. Deep Feature Extraction via VGG16
    vgg16 = get_vgg16_extractor()
    batch_input = np.expand_dims(processed_arr, axis=0) # shape: (1, 224, 224, 3)
    deep_features = vgg16.predict(batch_input, verbose=0) # shape: (1, 512)

    # 4. Standard Scaling using the fitted training scaler
    deep_features_scaled = scaler.transform(deep_features)

    # 5. SVM Prediction & Probability Estimation
    pred_idx = int(svm_model.predict(deep_features_scaled)[0])
    predicted_class = inv_label_map[pred_idx]

    # Probabilities via Platt Scaling
    if hasattr(svm_model, "predict_proba"):
        probs = svm_model.predict_proba(deep_features_scaled)[0]
    else:
        df = svm_model.decision_function(deep_features_scaled)[0]
        exp_df = np.exp(df - np.max(df))
        probs = exp_df / np.sum(exp_df)

    # Map probabilities to class names
    classes_in_order = [inv_label_map[i] for i in range(len(label_map))]
    prob_dict = {classes_in_order[i]: round(float(probs[i] * 100), 2) for i in range(len(classes_in_order))}
    top_confidence = prob_dict[predicted_class]

    return {
        'prediction': predicted_class,
        'confidence': top_confidence,
        'probabilities': prob_dict,
        'model_name': 'Pretrained VGG16 (512-dim) + RBF SVM',
        'disclaimer': bundle.get('disclaimer', 'Experimental statistical ML estimate.')
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict ABO Blood Group from Fingerprint Image using VGG16 + SVM Pipeline."
    )
    parser.add_argument(
        'image',
        nargs='?',
        default=None,
        help="Path to the input fingerprint image file."
    )
    parser.add_argument(
        '--image',
        dest='image_flag',
        default=None,
        help="Path to the input fingerprint image file."
    )
    parser.add_argument(
        '--model',
        default=MODEL_BUNDLE_PATH,
        help="Path to the trained model bundle (.pkl)."
    )

    args = parser.parse_args()
    image_path = args.image_flag or args.image

    if not image_path:
        # Default to a sample image from dataset if available
        sample_candidates = [
            os.path.join('dataset', 'dataset_blood_group', 'A+', 'cluster_0_1001.BMP'),
            os.path.join('dataset', 'dataset_blood_group', 'B+', 'cluster_2_10.BMP'),
            os.path.join('sample dataset', 'Sample 1.jpg')
        ]
        for sc in sample_candidates:
            if os.path.exists(sc):
                image_path = sc
                break

    if not image_path or not os.path.exists(image_path):
        print(f"[!] Error: Image file not found or not provided.")
        print(f"Usage: python predict.py <path_to_fingerprint_image>")
        sys.exit(1)

    print("=" * 80)
    print("      FINGERPRINT ABO BLOOD GROUP PREDICTION SYSTEM")
    print("=" * 80)
    print(f"[*] Input Image : {image_path}")

    try:
        with Image.open(image_path) as img:
            w, h = img.size
            mode = img.mode
            print(f"[*] Dimensions  : {w}x{h} | Mode: {mode}")
    except Exception as e:
        print(f"[!] Warning reading image metadata: {e}")

    result = predict_blood_group(image_path, model_path=args.model)

    print("\n" + "=" * 80)
    print(f"  PREDICTED BLOOD GROUP : {result['prediction']}")
    print(f"  CONFIDENCE SCORE      : {result['confidence']:.2f}%")
    print("=" * 80)

    print("\n  Class Probability Distribution Breakdown:")
    print(f"  {'Blood Group':<12} | {'Probability':<12} | {'Confidence Bar':<25}")
    print("  " + "-" * 55)
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    for bg, prob in sorted_probs:
        bar = "#" * int(prob / 4.0)
        marker = " <== [PREDICTED]" if bg == result['prediction'] else ""
        print(f"  {bg:<12} | {prob:10.2f}% | {bar:<25}{marker}")

    print("\n" + "-" * 80)
    print(f"[i] Statistical Note: {result['disclaimer']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
