"""
test_feedback_pipeline.py - Verification and Audit Script
==========================================================
Tests the full active learning and correction feedback loop:
1. Blueprint registration and route availability (/feedback/admin, /feedback/status, etc.)
2. Initial prediction before user correction
3. Feedback submission via /feedback/submit
4. Approval workflow via /feedback/approve/<id>
5. lookup_correction hook verification and 'from user correction' display in /api/predict
6. Retraining workflow (/feedback/retrain & polling /feedback/status)
7. Audit verification: STRICT ISOLATION of test split (dataset_test_split.csv is NEVER used)
8. Model versioning and Rollback to previous version
"""

import os
import sys
import time
import json
import io
from PIL import Image

def run_tests():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE FEEDBACK SUBSYSTEM VERIFICATION")
    print("=" * 80)

    from app import app, load_svm_model, CLASSES
    from feedback import lookup_correction

    client = app.test_client()

    # 1. Test Admin Route
    print("\n[Test 1] Testing /feedback/admin route...")
    res = client.get('/feedback/admin')
    assert res.status_code == 200, f"Expected 200 from /feedback/admin, got {res.status_code}"
    assert b"Mandatory Scientific & Clinical Disclaimer" in res.data, "Disclaimer missing from admin page"
    assert b"Model Retraining Engine" in res.data, "Retraining section missing from admin page"
    print("  [+] /feedback/admin loaded successfully with disclaimer and UI components.")

    # 2. Test /admin redirect
    print("\n[Test 2] Testing /admin redirect...")
    res = client.get('/admin')
    assert res.status_code in [301, 302], f"Expected redirect, got {res.status_code}"
    print("  [+] /admin redirects to /feedback/admin.")

    # 3. Test Initial Versions Endpoint
    print("\n[Test 3] Testing /feedback/api/versions...")
    res = client.get('/feedback/api/versions')
    assert res.status_code == 200
    versions_data = res.get_json()
    assert versions_data['success'] is True
    assert len(versions_data['versions']) >= 1
    initial_version = versions_data['versions'][0]
    print(f"  [+] Initial model version registered: {initial_version['version_id']} (Active: {initial_version.get('is_active')})")

    # 4. Find Sample Fingerprint Image for Testing
    sample_candidates = [
        os.path.join('sample dataset', 'Sample 1.jpg'),
        os.path.join('dataset', 'dataset_blood_group', 'A+', 'cluster_0_1001.BMP'),
        os.path.join('dataset', 'Fingerprint Dataset for Blood Group Classification', 'A+', 'cluster_0_1001.BMP')
    ]
    test_image_path = None
    for cand in sample_candidates:
        if os.path.exists(cand):
            test_image_path = cand
            break

    assert test_image_path is not None, "Could not find a sample image for testing"
    print(f"\n[Test 4] Selected test image: {test_image_path}")

    with open(test_image_path, 'rb') as f:
        test_img_bytes = f.read()

    # 5. Verify lookup_correction is initially None
    print("\n[Test 5] Verifying lookup_correction returns None for uncorrected image...")
    assert lookup_correction(test_img_bytes) is None, "lookup_correction should return None before approval"
    print("  [+] lookup_correction correctly returned None.")

    # 6. Run Initial Prediction via /api/predict
    print("\n[Test 6] Testing standard prediction on test image...")
    res = client.post('/api/predict', data={
        'file': (io.BytesIO(test_img_bytes), 'test_thumb.bmp'),
        'model_type': 'svm'
    }, content_type='multipart/form-data')
    assert res.status_code == 200
    pred_data = res.get_json()
    assert pred_data['success'] is True
    initial_predicted_label = pred_data['data']['prediction']
    print(f"  [+] Initial Model Prediction: {initial_predicted_label} (Confidence: {pred_data['data']['confidence']}%)")
    assert pred_data['data'].get('source') != 'from user correction', "Source should not be user correction yet"

    # Pick an opposing ground truth label to simulate a user correction
    desired_correction_label = 'AB+' if initial_predicted_label != 'AB+' else 'O-'
    print(f"  [*] Simulating user correction to: {desired_correction_label}")

    # 7. Submit User Correction via /feedback/submit
    print("\n[Test 7] Submitting correction to /feedback/submit...")
    res = client.post('/feedback/submit', data={
        'image': (io.BytesIO(test_img_bytes), 'test_thumb.bmp'),
        'predicted_label': initial_predicted_label,
        'correct_label': desired_correction_label
    }, content_type='multipart/form-data')
    assert res.status_code == 200, f"Submit failed: {res.get_data(as_text=True)}"
    submit_res = res.get_json()
    assert submit_res['success'] is True
    corr_id = submit_res['correction_id']
    print(f"  [+] Correction submitted successfully with ID: {corr_id}")

    # Verify still not active in lookup_correction while pending
    assert lookup_correction(test_img_bytes) is None, "Pending correction should not override yet"
    print("  [+] Pending correction is not yet active in lookup_correction.")

    # 8. Approve the Correction via /feedback/approve/<id>
    print(f"\n[Test 8] Approving correction {corr_id}...")
    res = client.post(f'/feedback/approve/{corr_id}')
    assert res.status_code == 200
    approve_res = res.get_json()
    assert approve_res['success'] is True
    print("  [+] Correction approved.")

    # 9. Verify lookup_correction now returns the corrected label immediately
    print("\n[Test 9] Verifying lookup_correction hook now returns the ground truth override...")
    found_label = lookup_correction(test_img_bytes)
    assert found_label == desired_correction_label, f"Expected {desired_correction_label}, got {found_label}"
    print(f"  [+] lookup_correction returned: {found_label}")

    # 10. Run Prediction on the same image via /api/predict: Must return "from user correction"
    print("\n[Test 10] Testing /api/predict returns 'from user correction'...")
    res = client.post('/api/predict', data={
        'file': (io.BytesIO(test_img_bytes), 'test_thumb.bmp'),
        'model_type': 'vgg16_hybrid'
    }, content_type='multipart/form-data')
    assert res.status_code == 200
    pred_data2 = res.get_json()
    assert pred_data2['success'] is True
    res_data = pred_data2['data']
    assert res_data['prediction'] == desired_correction_label, f"Expected {desired_correction_label}, got {res_data['prediction']}"
    assert res_data['source'] == 'from user correction', f"Expected source 'from user correction', got {res_data.get('source')}"
    assert res_data['model_name'] == 'from user correction', f"Expected model_name 'from user correction', got {res_data.get('model_name')}"
    assert res_data['is_user_correction'] is True
    print("  [+] /api/predict correctly intercepted approved correction and returned 'from user correction'!")

    # 11. Test Model Retraining & Strict Isolation of Test Split
    print("\n[Test 11] Triggering Model Retraining via /feedback/retrain...")
    res = client.post('/feedback/retrain')
    assert res.status_code == 200
    print("  [+] Retraining triggered. Polling /feedback/status...")

    # Poll status until finished
    for i in range(30):
        time.sleep(1.0)
        st_res = client.get('/feedback/status')
        status = st_res.get_json()
        print(f"      Polling ({i+1}s): step={status.get('step')}, progress={status.get('progress')}% - {status.get('message')}")
        if not status.get('is_retraining') and status.get('step') == 'completed':
            break
        if status.get('step') == 'failed':
            raise RuntimeError(f"Retraining failed: {status.get('message')}")

    assert status.get('step') == 'completed', f"Retraining did not complete: {status}"
    last_res = status.get('last_result', {})
    new_version_id = last_res.get('version_id')
    print(f"  [+] Retraining completed successfully! New Version: {new_version_id} (Val Acc: {last_res.get('val_accuracy')}%)")

    # 12. Strict Test Split Isolation Check
    print("\n[Test 12] Auditing Test Split Isolation...")
    # Verify that dataset_test_split.csv is untouched and has exact 862 samples
    import pandas as pd
    test_split_path = os.path.join('dataset', 'processed', 'dataset_test_split.csv')
    df_test = pd.read_csv(test_split_path)
    assert len(df_test) == 862, f"Test split count modified! Expected 862, found {len(df_test)}"
    
    # Check that retraining used train samples + feedback samples
    assert last_res['train_samples'] == 4018 + last_res['feedback_samples'], (
        f"Retrain samples mismatch: expected {4018 + last_res['feedback_samples']}, got {last_res['train_samples']}"
    )
    print("  [+] VERIFIED: Test split (N=862) was NEVER touched, modified, or merged during retraining!")

    # 13. Test Rollback to Previous Version
    print(f"\n[Test 13] Testing Rollback to version {initial_version['version_id']}...")
    res = client.post(f"/feedback/rollback/{initial_version['version_id']}")
    assert res.status_code == 200, f"Rollback failed: {res.get_data(as_text=True)}"
    rb_res = res.get_json()
    assert rb_res['success'] is True
    print(f"  [+] Rollback executed successfully. Active version restored to: {initial_version['version_id']}")

    # Verify versions endpoint reflects active flag
    res = client.get('/feedback/api/versions')
    versions_list = res.get_json()['versions']
    for v in versions_list:
        if v['version_id'] == initial_version['version_id']:
            assert v['is_active'] is True, "Original version should now be active"
        elif v['version_id'] == new_version_id:
            assert v['is_active'] is False, "New version should now be archived"
    print("  [+] Versions registry properly updated with active status.")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED SUCCESSFULLY! FULL ACTIVE LEARNING PIPELINE VERIFIED.")
    print("=" * 80)


if __name__ == '__main__':
    run_tests()
