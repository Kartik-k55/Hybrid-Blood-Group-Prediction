"""
Unit & Integration Tests for USB Fingerprint Scanner Architecture
==================================================================
Tests device enumeration, simulated sensor captures, hot-switching between scanners,
driver safety verification, and end-to-end ML prediction pipeline integration.
"""

import unittest
import io
import json
import base64
import numpy as np
from PIL import Image

from scanners import (
    get_scanner_manager,
    USBDetector,
    normalize_fingerprint_image,
    SimulatedScannerAdapter,
    Scanner1Adapter,
    Scanner2Adapter,
    ScannerStatus,
    ScannerNotConnectedError,
    DriverMissingError
)
from app import app


class TestScannerSubsystem(unittest.TestCase):

    def setUp(self):
        self.manager = get_scanner_manager()
        self.manager.select_scanner('simulated')
        self.client = app.test_client()

    def test_01_usb_detector_classification(self):
        """Verifies USB enumeration accurately detects devices and classifies non-biometrics."""
        devices = USBDetector.detect_devices()
        self.assertIsInstance(devices, list)
        
        # Verify non-biometric devices are marked as unsupported
        for dev in devices:
            self.assertIsNotNone(dev.name)
            self.assertIsNotNone(dev.vid)
            self.assertIsNotNone(dev.pid)
            if not dev.is_biometric_candidate:
                self.assertIn("Unsupported", dev.notes)

    def test_02_normalizer_canonical_output(self):
        """Verifies normalization to 512x512 grayscale with quality scores."""
        # Test with random 100x120 grayscale test array
        dummy_arr = np.random.randint(50, 220, (120, 100), dtype=np.uint8)
        pil_norm, png_bytes, data_url, quality = normalize_fingerprint_image(
            dummy_arr,
            target_size=(512, 512),
            maintain_aspect_ratio=True
        )

        self.assertEqual(pil_norm.size, (512, 512))
        self.assertEqual(pil_norm.mode, 'L')
        self.assertTrue(data_url.startswith("data:image/png;base64,"))
        self.assertGreaterEqual(quality['overall_score'], 0.0)
        self.assertLessEqual(quality['overall_score'], 100.0)

    def test_03_simulated_scanner_capture(self):
        """Verifies simulated scanner captures valid normalized biometric images."""
        sim_adapter = SimulatedScannerAdapter()
        self.assertTrue(sim_adapter.is_hardware_connected())
        self.assertTrue(sim_adapter.is_sdk_available())
        self.assertEqual(sim_adapter.get_status(), ScannerStatus.READY)

        capture_res = sim_adapter.capture()
        self.assertTrue(capture_res['success'])
        self.assertEqual(capture_res['normalized_dimensions'], (512, 512))
        self.assertTrue(capture_res['image_base64'].startswith("data:image/png;base64,"))
        self.assertIn('quality_score', capture_res)

    def test_04_scanner1_and_scanner2_safety(self):
        """Verifies physical scanner slots safely handle missing hardware and drivers."""
        s1 = Scanner1Adapter()
        s2 = Scanner2Adapter()

        self.assertEqual(s1.slot_number, 1)
        self.assertEqual(s2.slot_number, 2)

        # Unless hardware is plugged in, both should report disconnected without crashing
        if not s1.is_hardware_connected():
            self.assertEqual(s1.get_status(), ScannerStatus.DISCONNECTED)
            with self.assertRaises(ScannerNotConnectedError):
                s1.capture()

        if not s2.is_hardware_connected():
            self.assertEqual(s2.get_status(), ScannerStatus.DISCONNECTED)
            with self.assertRaises(ScannerNotConnectedError):
                s2.capture()

    def test_05_scanner_manager_switching(self):
        """Verifies hot-switching between scanners without web server restarts."""
        # Switch to Scanner 1
        res1 = self.manager.select_scanner('scanner_1')
        self.assertTrue(res1['success'])
        self.assertEqual(self.manager.active_scanner_id, 'scanner_1')

        # Switch to Scanner 2
        res2 = self.manager.select_scanner('scanner_2')
        self.assertTrue(res2['success'])
        self.assertEqual(self.manager.active_scanner_id, 'scanner_2')

        # Switch back to Simulated
        res_sim = self.manager.select_scanner('simulated')
        self.assertTrue(res_sim['success'])
        self.assertEqual(self.manager.active_scanner_id, 'simulated')

    def test_06_flask_scanner_api_endpoints(self):
        """Verifies Flask endpoints /api/scanners/list, /select, and /capture."""
        # 1. List
        res_list = self.client.get('/api/scanners/list')
        self.assertEqual(res_list.status_code, 200)
        data = res_list.get_json()['data']
        self.assertIn('slots', data)
        self.assertIn('active_scanner_id', data)
        self.assertIn('hardware_summary', data)

        # 2. Select
        res_select = self.client.post('/api/scanners/select', json={'scanner_id': 'simulated'})
        self.assertEqual(res_select.status_code, 200)
        self.assertEqual(res_select.get_json()['data']['active_scanner_id'], 'simulated')

        # 3. Capture
        res_cap = self.client.post('/api/scanners/capture', json={'scanner_id': 'simulated'})
        self.assertEqual(res_cap.status_code, 200)
        cap_data = res_cap.get_json()['data']
        self.assertTrue(cap_data['success'])
        self.assertEqual(cap_data['normalized_dimensions'], [512, 512])

    def test_07_scanned_image_prediction_pipeline(self):
        """Verifies sending captured scan through existing VGG16+SVM prediction pipeline."""
        # 1. Capture simulated fingerprint
        res_cap = self.client.post('/api/scanners/capture', json={'scanner_id': 'simulated'})
        img_b64 = res_cap.get_json()['data']['image_base64']

        # 2. Send base64 image into /api/predict
        res_pred = self.client.post('/api/predict', json={
            'image_base64': img_b64,
            'model_type': 'vgg16_hybrid'
        })
        self.assertEqual(res_pred.status_code, 200)
        pred_data = res_pred.get_json()['data']
        self.assertIn('prediction', pred_data)
        self.assertIn(pred_data['prediction'], ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-'])
        self.assertIn('confidence', pred_data)
        self.assertIn('probabilities', pred_data)

    def test_08_image_upload_preservation(self):
        """Verifies standard file upload is 100% preserved and fully independent."""
        # Generate dummy 120x120 PNG image
        img = Image.new('L', (120, 120), color=180)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        res_upload = self.client.post('/api/predict', data={
            'file': (buf, 'uploaded_test.png'),
            'model_type': 'vgg16_hybrid'
        }, content_type='multipart/form-data')

        self.assertEqual(res_upload.status_code, 200)
        data = res_upload.get_json()['data']
        self.assertIn('prediction', data)

    def test_09_cancel_scan_api(self):
        """Verifies POST /api/scanners/cancel endpoint and cancellation behavior."""
        res_cancel = self.client.post('/api/scanners/cancel')
        self.assertEqual(res_cancel.status_code, 200)
        res_json = res_cancel.get_json()
        self.assertTrue(res_json['success'])
        self.assertIn('Scan cancellation signal sent', res_json['data']['message'])

    def test_10_invalid_and_empty_images(self):
        """Verifies rejection of empty or corrupted image data."""
        # 1. Empty image bytes via file upload
        empty_buf = io.BytesIO(b"")
        res_empty_file = self.client.post('/api/predict', data={
            'file': (empty_buf, 'empty.png'),
            'model_type': 'vgg16_hybrid'
        }, content_type='multipart/form-data')
        self.assertEqual(res_empty_file.status_code, 400)
        self.assertFalse(res_empty_file.get_json()['success'])

        # 2. Corrupted base64
        res_corrupt_b64 = self.client.post('/api/predict', json={
            'image_base64': 'data:image/png;base64,NOT_A_VALID_IMAGE_DATA_CORRUPTED',
            'model_type': 'vgg16_hybrid'
        })
        self.assertEqual(res_corrupt_b64.status_code, 400)
        self.assertFalse(res_corrupt_b64.get_json()['success'])

        # 3. Microscopic image smaller than 16x16
        micro_img = Image.new('L', (8, 8), color=128)
        micro_buf = io.BytesIO()
        micro_img.save(micro_buf, format='PNG')
        micro_buf.seek(0)
        res_micro = self.client.post('/api/predict', data={
            'file': (micro_buf, 'tiny.png'),
            'model_type': 'vgg16_hybrid'
        }, content_type='multipart/form-data')
        self.assertEqual(res_micro.status_code, 400)
        self.assertFalse(res_micro.get_json()['success'])

    def test_11_driver_missing_status_label(self):
        """Verifies that when hardware is detected without SDK, exact label 'Device detected — capture integration required' is produced."""
        from scanners.usb_detector import USBDeviceInfo
        dev_mock = USBDeviceInfo(
            name="Mantra MFS100 Biometric Device",
            device_id="USB\\VID_27D0&PID_1001\\01",
            vid="27D0",
            pid="1001",
            manufacturer="Mantra",
            status="OK",
            present=True,
            is_biometric_candidate=True,
            vendor_name="Mantra Softech",
            model_name="MFS100 Optical Fingerprint Scanner",
            sdk_name="Mantra MFS100 SDK",
            device_type="Optical Platen Scanner",
            notes="Biometric sensor match"
        )
        s1 = Scanner1Adapter()
        # Mock _get_matched_device to return dev_mock and is_sdk_available to return False
        orig_matched = s1._get_matched_device
        orig_sdk = s1.is_sdk_available
        try:
            s1._get_matched_device = lambda force_refresh=False: dev_mock
            s1.is_sdk_available = lambda: False
            telemetry = s1.get_telemetry()
            self.assertEqual(telemetry['status_label'], "Device detected — capture integration required")
        finally:
            s1._get_matched_device = orig_matched
            s1.is_sdk_available = orig_sdk

    def test_12_simulated_mode_explicit_label(self):
        """Verifies simulated mode explicitly identifies as virtual demo and does not claim physical hardware."""
        sim = SimulatedScannerAdapter()
        telemetry = sim.get_telemetry()
        self.assertTrue(telemetry['is_simulated'])
        self.assertIn("Demo / Simulated", telemetry['display_name'])
        self.assertEqual(telemetry['vid'], "VIRTUAL")
        self.assertEqual(telemetry['pid'], "DEMO")
        self.assertIn("No physical hardware", telemetry['integration_notes'])


if __name__ == '__main__':
    unittest.main()
