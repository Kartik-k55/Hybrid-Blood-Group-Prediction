import os
import sys
import glob
import json
import numpy as np
from PIL import Image

# Ensure UTF-8 output encoding for windows consoles
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from app import run_resnet_inference, run_hybrid_resnet_svm_inference, run_ensemble_inference

POSITIVE_CLASSES = ['A+', 'B+', 'AB+', 'O+']
DATASET_DIR = os.path.join('dataset', 'dataset_blood_group')

def test_positive_bloodgroups():
    print("=" * 80)
    print("   RAKTA MITRA: PREDICTING FINGERPRINTS FOR POSITIVE BLOOD GROUPS (Rh+)")
    print("=" * 80)
    
    results_summary = []
    
    for bg in POSITIVE_CLASSES:
        bg_folder = os.path.join(DATASET_DIR, bg)
        if not os.path.exists(bg_folder):
            print(f"[!] Folder for {bg} not found, skipping.")
            continue
            
        # Get sample images
        imgs = glob.glob(os.path.join(bg_folder, '*.BMP')) + \
               glob.glob(os.path.join(bg_folder, '*.bmp')) + \
               glob.glob(os.path.join(bg_folder, '*.jpg'))
               
        if not imgs:
            print(f"[!] No images found for {bg}")
            continue
            
        sample_path = imgs[0]
        print(f"\n" + "-" * 80)
        print(f"[*] Processing Sample for Ground Truth Blood Group: {bg}")
        print(f"    File: {sample_path}")
        
        with Image.open(sample_path) as img:
            # 1. ResNet Model Prediction
            resnet_result = run_resnet_inference(img)
            
            # 2. Hybrid ResNet-SVM Prediction
            hybrid_result = run_hybrid_resnet_svm_inference(img)
            
            # 3. Ensemble Prediction
            ensemble_result = run_ensemble_inference(img)
            
            bio = resnet_result['biometrics']
            
            print(f"\n  [Biometric Analysis]")
            print(f"    - Dimensions         : {bio['dimensions']}")
            print(f"    - Ridge Density Score: {bio['ridge_density_score']}")
            print(f"    - Mean Intensity     : {bio['mean_intensity']}")
            print(f"    - Contrast Std Dev   : {bio['contrast_std']}")
            
            print(f"\n  [Model Predictions]")
            print(f"    - ResNet Deep CNN       : Predicted = {resnet_result['prediction']} | Confidence = {resnet_result['confidence']}%")
            print(f"    - Hybrid ResNet-SVM     : Predicted = {hybrid_result['prediction']} | Confidence = {hybrid_result['confidence']}%")
            print(f"    - Unified Ensemble      : Predicted = {ensemble_result['prediction']} | Confidence = {ensemble_result['confidence']}%")
            print(f"    - Rh Factor Detected    : {'Positive (+)' if '+' in ensemble_result['prediction'] else 'Negative (-)'}")
            
            print(f"\n  [Top Probabilities Breakdown]")
            sorted_probs = sorted(ensemble_result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:4]
            for blood_type, prob in sorted_probs:
                bar = "#" * int(prob / 2.5)
                print(f"    {blood_type:>4}: {prob:6.2f}% | {bar}")
                
            results_summary.append({
                'ground_truth': bg,
                'file': os.path.basename(sample_path),
                'resnet_pred': resnet_result['prediction'],
                'hybrid_pred': hybrid_result['prediction'],
                'ensemble_pred': ensemble_result['prediction'],
                'confidence': ensemble_result['confidence'],
                'rh_factor': 'Positive (Rh+)' if '+' in ensemble_result['prediction'] else 'Negative (Rh-)'
            })

    print("\n" + "=" * 80)
    print("                      POSITIVE BLOOD GROUP SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Target':<8} | {'File Name':<20} | {'ResNet':<8} | {'Hybrid SVM':<10} | {'Ensemble':<10} | {'Rh Status':<12}")
    print("-" * 80)
    for r in results_summary:
        print(f"{r['ground_truth']:<8} | {r['file']:<20} | {r['resnet_pred']:<8} | {r['hybrid_pred']:<10} | {r['ensemble_pred']:<10} | {r['rh_factor']:<12}")
    print("=" * 80)

if __name__ == '__main__':
    test_positive_bloodgroups()
