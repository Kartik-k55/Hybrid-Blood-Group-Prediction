"""
Comprehensive Automated Test Suite for Fingerprint Blood Group Classification
=============================================================================
Tests:
1. Dataset loading & manifest integrity
2. 8-class label mapping
3. Preprocessing pipeline consistency (CLAHE, grayscale, resizing)
4. Model loading (ResNet-50, Hybrid ResNet-SVM, Classical SVM, Hybrid VGG16-SVM)
5. Single-image prediction on EVERY class (A+, A-, AB+, AB-, B+, B-, O+, O-)
6. Invalid image rejection (blank, extreme blur, noise)
7. Flask routes & API endpoints
8. Image upload handling
9. Scanner simulation & normalizer
10. Prediction result structure & 8-class probability distribution
"""

import os
import io
import sys
import json
import unittest
import numpy as np
import pandas as pd
from PIL import Image

# Output UTF-8 handling
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import (
    app,
    CLASSES,
    load_resnet_model,
    load_hybrid_svm_model,
    load_svm_model,
    load_hybrid_vgg16_svm_model,
    run_ensemble_inference,
    run_hybrid_vgg16_svm_inference,
    run_hybrid_resnet_svm_inference,
    run_svm_inference,
    run_resnet_inference,
    validate_fingerprint_image
)
from preprocessing import (
    preprocess_fingerprint,
    apply_clahe_enhancement,
    TARGET_IMAGE_SIZE_VGG16
)
from scanners import get_scanner_manager


class TestFingerprintBloodGroupPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.test_csv_path = os.path.join('dataset', 'processed', 'dataset_test_split.csv')
        cls.test_df = pd.read_csv(cls.test_csv_path)

    def test_01_dataset_and_manifests(self):
        """1. Verify dataset split manifests and exact sample counts."""
        self.assertTrue(os.path.exists('dataset/processed/dataset_train_split.csv'))
        self.assertTrue(os.path.exists('dataset/processed/dataset_val_split.csv'))
        self.assertTrue(os.path.exists('dataset/processed/dataset_test_split.csv'))
        
        train_df = pd.read_csv('dataset/processed/dataset_train_split.csv')
        val_df = pd.read_csv('dataset/processed/dataset_val_split.csv')
        test_df = pd.read_csv('dataset/processed/dataset_test_split.csv')

        self.assertEqual(len(train_df), 4018)
        self.assertEqual(len(val_df), 861)
        self.assertEqual(len(test_df), 862)
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), 5741)

    def test_02_eight_class_mapping(self):
        """2. Verify 8-class blood group definitions."""
        expected_classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
        self.assertEqual(CLASSES, expected_classes)
        self.assertEqual(len(CLASSES), 8)

        # Verify label mapping json
        with open('models/label_mapping.json', 'r') as f:
            mapping = json.load(f)
        for c in expected_classes:
            self.assertIn(c, mapping)

    def test_03_preprocessing(self):
        """3. Verify preprocessing pipeline consistency."""
        sample_path = self.test_df['filepath'].iloc[0]
        img = Image.open(sample_path)
        
        # Test CLAHE
        gray = np.array(img.convert('L'), dtype=np.uint8)
        enhanced = apply_clahe_enhancement(gray)
        self.assertEqual(enhanced.shape, gray.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

        # Test model tensor preprocess
        tensor_vgg = preprocess_fingerprint(img, target_size=(224, 224), model_type='vgg16')
        self.assertEqual(tensor_vgg.shape, (224, 224, 3))

        tensor_res = preprocess_fingerprint(img, target_size=(256, 256), model_type='resnet')
        self.assertEqual(tensor_res.shape, (256, 256, 3))

    def test_04_model_loading(self):
        """4. Verify that all trained models load successfully."""
        resnet = load_resnet_model()
        self.assertIsNotNone(resnet, "ResNet model failed to load")

        hyb_res = load_hybrid_svm_model()
        self.assertIsNotNone(hyb_res, "Hybrid ResNet-SVM model failed to load")

        svm = load_svm_model()
        self.assertIsNotNone(svm, "Classical SVM model failed to load")

        hyb_vgg = load_hybrid_vgg16_svm_model()
        self.assertIsNotNone(hyb_vgg, "Hybrid VGG16-SVM model failed to load")

    def test_05_predict_every_class(self):
        """5. Test prediction on at least one real image from EVERY one of the 8 classes."""
        print("\n  Testing inference on at least one sample from each of the 8 classes:")
        for bg in CLASSES:
            class_samples = self.test_df[self.test_df['class_name'] == bg]
            self.assertGreater(len(class_samples), 0, f"No test samples found for class {bg}")
            
            sample_path = class_samples.iloc[0]['filepath']
            img = Image.open(sample_path)
            
            # Run ensemble inference
            res = run_ensemble_inference(img)
            pred = res['prediction']
            conf = res['confidence']
            probs = res['probabilities']

            print(f"    Class [{bg:4s}] -> Predicted: [{pred:4s}] | Confidence: {conf:5.2f}% | Top-3: {sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]}")
            
            self.assertIn(pred, CLASSES, f"Predicted class {pred} not in known classes")
            self.assertGreater(conf, 0.0)
            self.assertEqual(len(probs), 8)
            for c in CLASSES:
                self.assertIn(c, probs)

    def test_06_invalid_image_rejection(self):
        """6. Verify rejection of blank, severely blurred, and non-fingerprint images."""
        # A. Blank White Image
        blank_white = Image.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 255)
        with self.assertRaises(ValueError) as ctx:
            validate_fingerprint_image(blank_white)
        self.assertIn("Unable to classify this fingerprint", str(ctx.exception))

        # B. Blank Black Image
        blank_black = Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8))
        with self.assertRaises(ValueError) as ctx:
            validate_fingerprint_image(blank_black)
        self.assertIn("Unable to classify this fingerprint", str(ctx.exception))

        # C. Severely Blurred Image
        import cv2
        blurred_noise = Image.fromarray(cv2.GaussianBlur(np.random.randint(100, 150, (100, 100), dtype=np.uint8), (35, 35), 10))
        with self.assertRaises(ValueError) as ctx:
            validate_fingerprint_image(blurred_noise)
        self.assertIn("Unable to classify this fingerprint", str(ctx.exception))

        # D. Test API rejection via HTTP endpoint
        buf = io.BytesIO()
        blank_white.save(buf, format='PNG')
        buf.seek(0)
        resp = self.client.post('/api/predict', data={'file': (buf, 'blank.png')}, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data['success'])
        self.assertEqual(data['error'], "Unable to classify this fingerprint. Please provide a clear fingerprint image.")

    def test_07_flask_routes(self):
        """7. Verify primary Flask endpoints."""
        # Index
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)

        # Samples
        resp = self.client.get('/api/samples')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('samples', data)
        self.assertGreater(len(data['samples']), 0)

        # Scanners list
        resp = self.client.get('/api/scanners/list')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['success'])

        # Secure status
        resp = self.client.get('/api/secure/status')
        self.assertEqual(resp.status_code, 200)

    def test_08_image_upload_and_prediction(self):
        """8. Test prediction via multipart file upload."""
        sample_path = self.test_df['filepath'].iloc[0]
        with open(sample_path, 'rb') as f:
            img_bytes = f.read()

        buf = io.BytesIO(img_bytes)
        resp = self.client.post('/api/predict', data={'file': (buf, 'fingerprint.bmp'), 'model_type': 'ensemble'}, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)
        res = resp.get_json()
        self.assertTrue(res['success'])
        self.assertIn('prediction', res['data'])
        self.assertIn('probabilities', res['data'])
        self.assertEqual(len(res['data']['probabilities']), 8)
        self.assertIn('model_version', res['data'])

    def test_09_scanner_simulation(self):
        """9. Test simulated scanner capture and normalization."""
        mgr = get_scanner_manager()
        mgr.select_scanner('simulated')
        capture_res = mgr.capture_fingerprint(scanner_id='simulated', timeout_seconds=5.0)
        self.assertTrue(capture_res['success'])
        self.assertIn('image_base64', capture_res)
        self.assertEqual(capture_res['normalized_dimensions'], (512, 512))
        self.assertEqual(capture_res['image_pil'].size, (512, 512))

        # Feed capture to /api/predict
        resp = self.client.post('/api/predict', json={'image_base64': capture_res['image_base64'], 'model_type': 'ensemble'})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['success'])
        self.assertIn(data['data']['prediction'], CLASSES)

    def test_10_prediction_result_structure(self):
        """10. Verify full prediction payload structure."""
        sample_path = self.test_df['filepath'].iloc[0]
        resp = self.client.post('/api/predict', json={'sample_path': sample_path, 'model_type': 'ensemble'})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()['data']

        required_keys = ['prediction', 'confidence', 'probabilities', 'biometrics', 'model_name', 'model_version', 'disclaimer']
        for k in required_keys:
            self.assertIn(k, data, f"Missing required payload key: {k}")

        self.assertIn(data['prediction'], CLASSES)
        self.assertIsInstance(data['confidence'], float)
        self.assertEqual(len(data['probabilities']), 8)


if __name__ == '__main__':
    unittest.main()
