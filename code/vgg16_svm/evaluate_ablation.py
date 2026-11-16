"""
===============================================================================
STANDALONE ABLATION EVALUATOR & CALIBRATION INSPECTOR
===============================================================================
"""

import os
import sys
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from code.vgg16_svm.prevalence import REAL_WORLD_PREVALENCE, compare_distribution_alignment
except ImportError:
    from prevalence import REAL_WORLD_PREVALENCE, compare_distribution_alignment

MODEL_ARTIFACT_PATH = os.path.join('test', 'model_blood_group_detection_hybrid_vgg16_svm.pkl')

def inspect_saved_pipeline():
    if not os.path.exists(MODEL_ARTIFACT_PATH):
        print(f"Artifact not found at {MODEL_ARTIFACT_PATH}. Please run train_pipeline.py first.")
        return
        
    with open(MODEL_ARTIFACT_PATH, 'rb') as f:
        data = pickle.load(f)
        
    print("=" * 70)
    print("HYBRID VGG16-SVM SAVED PIPELINE ARTIFACT SUMMARY")
    print("=" * 70)
    print(f"Model Name:      {data.get('model_name')}")
    print(f"SVM Estimator:   {data.get('svm')}")
    print(f"PCA Components:  {data.get('pca').n_components_}")
    print(f"Classes (8):     {data.get('classes')}")
    print("\nRecorded Test & Validation Metrics:")
    metrics = data.get('metrics', {})
    for k, v in metrics.items():
        print(f"  - {k:<30}: {v}")
    print("=" * 70)

if __name__ == '__main__':
    inspect_saved_pipeline()
