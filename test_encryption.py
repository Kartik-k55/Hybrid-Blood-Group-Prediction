"""
Comprehensive Verification and Test Suite for AES-256-GCM Encryption
=====================================================================
Tests:
1. Key generation & validation across formats (Hex, Base64, raw 32-byte)
2. String, Dictionary/JSON, and Bytes encryption -> decryption roundtrips
3. File encryption -> decryption roundtrips using real fingerprint sample images
4. Nonce freshness (distinct ciphertexts for same plaintext)
5. Tamper detection and authentication failure (AES-GCM integrity)
6. Associated data (AAD) mismatch verification
7. Error handling for missing/invalid keys
8. Flask API encryption endpoints integration
"""

import os
import json
import base64
import tempfile
import unittest
from io import BytesIO
from PIL import Image

# Import encryption module
import encryption
from encryption import (
    generate_key,
    get_aes_key,
    encrypt_data,
    decrypt_data,
    encrypt_bytes,
    decrypt_bytes,
    encrypt_file,
    decrypt_file,
    EncryptionError,
    MissingKeyError,
    InvalidKeyError,
    DecryptionError
)


class TestAES256GCMEncryption(unittest.TestCase):
    def setUp(self):
        # Generate a test 256-bit key
        self.test_key_hex = generate_key('hex')
        self.test_key_b64 = generate_key('base64')
        self.test_key_raw = generate_key('raw')
        # Set environment variable for default key resolution
        os.environ['AES_KEY'] = self.test_key_hex

    def test_key_generation(self):
        """Test key generator output format and byte lengths."""
        hex_key = generate_key('hex')
        self.assertEqual(len(hex_key), 64)
        self.assertEqual(len(bytes.fromhex(hex_key)), 32)

        b64_key = generate_key('base64')
        self.assertEqual(len(base64.b64decode(b64_key)), 32)

        raw_key = generate_key('raw')
        self.assertEqual(len(raw_key), 32)

    def test_key_validation(self):
        """Test key format detection and invalid key exceptions."""
        # Valid keys
        self.assertEqual(len(get_aes_key(self.test_key_hex)), 32)
        self.assertEqual(len(get_aes_key(self.test_key_b64)), 32)
        self.assertEqual(len(get_aes_key(self.test_key_raw)), 32)

        # Invalid key lengths
        with self.assertRaises(InvalidKeyError):
            get_aes_key("short_key")
        with self.assertRaises(InvalidKeyError):
            get_aes_key("x" * 50)

        # Missing key when env var is stripped
        old_key = os.environ.pop('AES_KEY', None)
        try:
            with self.assertRaises(MissingKeyError):
                get_aes_key()
        finally:
            if old_key:
                os.environ['AES_KEY'] = old_key

    def test_string_roundtrip(self):
        """Verify: string -> encrypt -> decrypt -> exact original string."""
        sample_texts = [
            "Patient #1042: Blood Group A+, Ridge Density: 14.8",
            "Sensitive Biometric Minutiae Coordinates: [(120, 85), (204, 190)]",
            "Special characters: äöü!@#$%^&*()_+{}[]:;<>,.?/|~`",
            "" # Empty string
        ]
        for text in sample_texts:
            ciphertext = encrypt_data(text)
            self.assertNotEqual(text, ciphertext)
            decrypted = decrypt_data(ciphertext)
            self.assertEqual(text, decrypted)

    def test_json_dict_roundtrip(self):
        """Verify: dict/list -> encrypt -> decrypt -> original data structure."""
        biometrics_data = {
            'patient_id': 'PATIENT-9982',
            'predicted_blood_group': 'O+',
            'confidence': 98.75,
            'ridge_spacing_px': 9.42,
            'spatial_minutiae_zones': {
                'top_left': 0.142,
                'top_right': 0.215,
                'bottom_left': 0.089,
                'bottom_right': 0.194,
                'center_core': 0.358
            },
            'models_evaluated': ['ResNet-34', 'Hybrid VGG16-SVM', 'Bayesian Calibrator']
        }

        ciphertext = encrypt_data(biometrics_data)
        decrypted = decrypt_data(ciphertext, return_json=True)
        self.assertEqual(biometrics_data, decrypted)

    def test_bytes_roundtrip(self):
        """Verify: raw bytes -> encrypt -> decrypt -> exact original bytes."""
        raw_bytes = os.urandom(1024 * 64) # 64 KB of random data
        ciphertext = encrypt_bytes(raw_bytes)
        decrypted = decrypt_bytes(ciphertext)
        self.assertEqual(raw_bytes, decrypted)

    def test_nonce_freshness(self):
        """Verify that encrypting the exact same plaintext twice produces distinct ciphertexts."""
        plaintext = "Identical biometric fingerprint template string"
        cipher1 = encrypt_data(plaintext)
        cipher2 = encrypt_data(plaintext)
        self.assertNotEqual(cipher1, cipher2)
        self.assertEqual(decrypt_data(cipher1), plaintext)
        self.assertEqual(decrypt_data(cipher2), plaintext)

    def test_tamper_detection_and_authentication(self):
        """Verify that any modification to ciphertext or tag causes decryption/authentication failure."""
        plaintext = "Classified Patient Biometric Data"
        ciphertext = encrypt_data(plaintext)
        raw_cipher = base64.b64decode(ciphertext)

        # Flip one bit in the middle of ciphertext
        tampered_byte = bytes([raw_cipher[20] ^ 0x01])
        tampered_cipher = raw_cipher[:20] + tampered_byte + raw_cipher[21:]
        tampered_b64 = base64.b64encode(tampered_cipher).decode('ascii')

        with self.assertRaises(DecryptionError):
            decrypt_data(tampered_b64)

    def test_associated_data_aad(self):
        """Verify authenticated associated data (AAD) protection."""
        plaintext = "Blood Group B+ Confirmed"
        aad_correct = "User:Dr.Sharma|Hospital:Apollo"
        aad_wrong = "User:Dr.Sharma|Hospital:Fortis"

        cipher = encrypt_data(plaintext, associated_data=aad_correct)
        
        # Correct AAD succeeds
        decrypted = decrypt_data(cipher, associated_data=aad_correct)
        self.assertEqual(decrypted, plaintext)

        # Mismatched AAD fails authentication
        with self.assertRaises(DecryptionError):
            decrypt_data(cipher, associated_data=aad_wrong)

    def test_file_encryption_decryption(self):
        """Verify file encryption -> decryption on an actual sample image."""
        sample_img_path = None
        for root, _, files in os.walk('.'):
            for f in files:
                if f.lower().endswith(('.bmp', '.jpg', '.png')) and ('sample' in root or 'test' in root or 'dataset' in root):
                    sample_img_path = os.path.join(root, f)
                    break
            if sample_img_path:
                break

        if sample_img_path:
            with tempfile.NamedTemporaryFile(suffix='.enc', delete=False) as enc_temp:
                enc_path = enc_temp.name
            with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as dec_temp:
                dec_path = dec_temp.name

            try:
                # Encrypt file
                encrypt_file(sample_img_path, enc_path)
                self.assertTrue(os.path.exists(enc_path))
                
                # Decrypt file
                decrypt_file(enc_path, dec_path)
                self.assertTrue(os.path.exists(dec_path))

                # Verify byte equality
                with open(sample_img_path, 'rb') as orig_f, open(dec_path, 'rb') as dec_f:
                    self.assertEqual(orig_f.read(), dec_f.read())
            finally:
                if os.path.exists(enc_path):
                    os.remove(enc_path)
                if os.path.exists(dec_path):
                    os.remove(dec_path)


class TestFlaskEncryptionEndpoints(unittest.TestCase):
    def setUp(self):
        os.environ['AES_KEY'] = generate_key('hex')
        from app import app
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_status_endpoint(self):
        """Test /api/secure/status returns ready status without leaking secret key value."""
        res = self.client.get('/api/secure/status')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get('aes_ready'))
        self.assertEqual(data.get('algorithm'), 'AES-256-GCM')
        # Ensure actual raw secret key is never leaked in the response
        secret_key_val = os.environ.get('AES_KEY')
        if secret_key_val:
            self.assertNotIn(secret_key_val, str(data))

    def test_secure_encrypt_decrypt_endpoints(self):
        """Test API endpoints for secure transmission."""
        payload = {
            'donor': 'DONOR-772',
            'fingerprint_density': 18.2,
            'prediction': 'AB+'
        }
        
        # 1. Encrypt via API
        enc_res = self.client.post('/api/secure/encrypt', json={'data': payload})
        self.assertEqual(enc_res.status_code, 200)
        enc_data = enc_res.get_json()
        self.assertTrue(enc_data.get('success'))
        ciphertext = enc_data.get('encrypted')
        self.assertIsNotNone(ciphertext)

        # 2. Decrypt via API
        dec_res = self.client.post('/api/secure/decrypt', json={'encrypted': ciphertext})
        self.assertEqual(dec_res.status_code, 200)
        dec_data = dec_res.get_json()
        self.assertTrue(dec_data.get('success'))
        self.assertEqual(dec_data.get('data'), payload)

    def test_secure_vault_records_save_and_retrieve(self):
        """Test at-rest secure vault record storage and decryption."""
        record_data = {
            'patient_id': 'TEST_PATIENT_42',
            'data': {
                'blood_group': 'O+',
                'confidence': 99.4,
                'minutiae_count': 48
            }
        }
        # Save record
        save_res = self.client.post('/api/secure/save-record', json=record_data)
        self.assertEqual(save_res.status_code, 200)
        save_json = save_res.get_json()
        self.assertTrue(save_json.get('success'))
        record_id = save_json.get('record_id')

        # List records
        list_res = self.client.get('/api/secure/records')
        self.assertEqual(list_res.status_code, 200)
        records = list_res.get_json().get('records', [])
        self.assertTrue(any(r['record_id'] == record_id for r in records))

        # Retrieve & Decrypt record
        get_res = self.client.get(f'/api/secure/records/{record_id}')
        self.assertEqual(get_res.status_code, 200)
        get_json = get_res.get_json()
        self.assertTrue(get_json.get('success'))
        self.assertEqual(get_json['record']['patient_id'], 'TEST_PATIENT_42')
        self.assertEqual(get_json['record']['data']['blood_group'], 'O+')

    def test_predict_with_encrypted_image(self):
        """Test sending an AES-256-GCM encrypted fingerprint image to /api/predict."""
        # Find a sample image
        sample_path = None
        for root, _, files in os.walk('.'):
            for f in files:
                if f.lower().endswith(('.bmp', '.jpg', '.png')) and ('sample' in root or 'test' in root or 'dataset' in root):
                    sample_path = os.path.join(root, f)
                    break
            if sample_path:
                break

        if sample_path:
            with open(sample_path, 'rb') as f:
                raw_bytes = f.read()
            
            # Encrypt raw image bytes with AES-256-GCM
            encrypted_b64 = encrypt_data(base64.b64encode(raw_bytes).decode('ascii'))

            # Send encrypted image to prediction endpoint
            res = self.client.post('/api/predict', json={
                'encrypted_image': encrypted_b64,
                'model_type': 'vgg16_hybrid',
                'encrypt_response': True
            })
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertTrue(data.get('success'))
            self.assertTrue(data.get('encrypted'))
            self.assertIn('encrypted_payload', data)

            # Decrypt response
            decrypted_result = decrypt_data(data['encrypted_payload'], return_json=True)
            self.assertIn('prediction', decrypted_result)
            self.assertIn('confidence', decrypted_result)
            self.assertIn(decrypted_result['prediction'], ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
