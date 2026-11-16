"""
===============================================================================
MASTER HYBRID ML+DL VGG16-SVM TRAINING & EVALUATION PIPELINE
===============================================================================

Key Features Implemented:
1. Stratified 70 / 15 / 15 Train / Validation / Test Splitting.
2. Fine-Tuning VGG16 (Block5 unfreezing, lr=1e-5, EarlyStopping, ReduceLROnPlateau).
3. Dermatoglyphic Ridge-Pattern Classification (Loop, Whorl, Arch, Composite) & Domain Priors.
4. Concatenated Multi-Modal Feature Vector (Deep CNN 512-dim + Ridge Biometrics).
5. PCA Dimensionality Reduction (retaining 95% cumulative variance).
6. 5-Fold Stratified Cross-Validation for SVM robustness.
7. Hyperparameter Tuning via GridSearchCV (C, gamma, kernel).
8. Bayesian Real-World Population Prevalence Calibration (Rh/ABO adjustment).
9. Full Multi-Model Ablation Study & Calibration Analysis.
10. Permutation Feature Importance Analysis.
"""

import os
import sys
import glob
import time
import pickle
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.inspection import permutation_importance

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import our custom modular components
try:
    from code.vgg16_svm.ridge_patterns import (
        classify_ridge_pattern,
        extract_ridge_domain_feature_vector,
        compute_dermatoglyphic_prior_scores,
        DEFAULT_PRIOR_MULTIPLIERS
    )
    from code.vgg16_svm.prevalence import (
        BayesianPrevalenceCalibrator,
        compare_distribution_alignment,
        REAL_WORLD_PREVALENCE
    )
    from code.vgg16_svm.vgg16_extractor import (
        preprocess_image_for_vgg16,
        apply_training_augmentation,
        train_and_finetune_vgg16,
        build_vgg16_classifier,
        build_vgg16_feature_extractor
    )
except ImportError:
    from ridge_patterns import (
        classify_ridge_pattern,
        extract_ridge_domain_feature_vector,
        compute_dermatoglyphic_prior_scores,
        DEFAULT_PRIOR_MULTIPLIERS
    )
    from prevalence import (
        BayesianPrevalenceCalibrator,
        compare_distribution_alignment,
        REAL_WORLD_PREVALENCE
    )
    from vgg16_extractor import (
        preprocess_image_for_vgg16,
        apply_training_augmentation,
        train_and_finetune_vgg16,
        build_vgg16_classifier,
        build_vgg16_feature_extractor
    )

# Output UTF-8 handling for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATASET_DIR = os.path.join('dataset', 'dataset_blood_group')
SAVE_MODEL_DIR = 'test'
HYBRID_VGG16_SAVE_PATH = os.path.join(SAVE_MODEL_DIR, 'model_blood_group_detection_hybrid_vgg16_svm.pkl')
VGG16_KERAS_SAVE_PATH = os.path.join(SAVE_MODEL_DIR, 'model_vgg16_finetuned_backbone.h5')

CLASSES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

def load_dataset_records(max_samples_per_class=350, random_seed=42):
    """
    Discover images across all 8 ABO/Rh blood group folders.
    """
    print(f"[*] Discovering dataset images from: {DATASET_DIR}")
    records = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_folder = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_folder):
            print(f"  [!] Warning: Folder {class_folder} not found, skipping.")
            continue
            
        file_list = []
        for ext in ['*.BMP', '*.bmp', '*.jpg', '*.jpeg', '*.png']:
            file_list.extend(glob.glob(os.path.join(class_folder, ext)))
        file_list = sorted(list(set(file_list)))
        
        np.random.seed(random_seed)
        n_select = min(len(file_list), max_samples_per_class)
        chosen = np.random.choice(file_list, n_select, replace=False)
        
        for fpath in chosen:
            records.append({
                'path': fpath,
                'class_idx': class_idx,
                'class_name': class_name
            })
        print(f"  [+] Class '{class_name}': selected {n_select} samples (from {len(file_list)} found)")
        
    print(f"[OK] Total dataset samples collected: {len(records)}\n")
    return records

def perform_stratified_70_15_15_split(records, random_seed=42):
    """
    Perform exact 70% Train, 15% Validation, 15% Test stratified split.
    """
    paths = [r['path'] for r in records]
    labels = np.array([r['class_idx'] for r in records], dtype=np.int64)
    
    # 1. Split Train (70%) vs Temp (30%)
    train_paths, temp_paths, y_train, y_temp = train_test_split(
        paths, labels, test_size=0.30, random_state=random_seed, stratify=labels
    )
    
    # 2. Split Temp into Val (15%) and Test (15%)
    val_paths, test_paths, y_val, y_test = train_test_split(
        temp_paths, y_temp, test_size=0.50, random_state=random_seed, stratify=y_temp
    )
    
    total = len(records)
    print("=" * 75)
    print("STRATIFIED DATASET SPLIT (70% Train / 15% Validation / 15% Test)")
    print("=" * 75)
    print(f"[*] Training Split:   {len(train_paths):4d} samples ({len(train_paths)/total*100:.1f}%)")
    print(f"[*] Validation Split: {len(val_paths):4d} samples ({len(val_paths)/total*100:.1f}%)")
    print(f"[*] Testing Split:    {len(test_paths):4d} samples ({len(test_paths)/total*100:.1f}%)")
    print("=" * 75)
    
    return train_paths, y_train, val_paths, y_val, test_paths, y_test

def extract_features_for_split(paths, feature_extractor, augment=False, batch_size=32):
    """
    Extract multi-modal feature representations:
    1. VGG16 Deep Feature Vector (512-dim GAP)
    2. Ridge Pattern Biometric Vector (15-dim)
    3. Rule-Based Dermatoglyphic Prior Vector (8-dim)
    """
    deep_vectors = []
    ridge_vectors = []
    prior_dicts_list = []
    
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        batch_vgg_inputs = []
        
        for fpath in batch_paths:
            with Image.open(fpath) as pil_img:
                # Ridge domain features
                ridge_vec, pattern_name, pattern_probs, feats = extract_ridge_domain_feature_vector(pil_img)
                prior_dict = compute_dermatoglyphic_prior_scores(pattern_probs)
                
                # Preprocess for VGG16
                vgg_px = preprocess_image_for_vgg16(pil_img)
                if augment:
                    vgg_px = apply_training_augmentation(vgg_px)
                    
                batch_vgg_inputs.append(vgg_px)
                ridge_vectors.append(ridge_vec)
                prior_dicts_list.append(prior_dict)
                
        batch_tensor = np.array(batch_vgg_inputs, dtype=np.float32)
        deep_embeddings = feature_extractor.predict(batch_tensor, batch_size=len(batch_tensor), verbose=0)
        deep_vectors.extend(deep_embeddings)
        
    return (
        np.array(deep_vectors, dtype=np.float32),
        np.array(ridge_vectors, dtype=np.float32),
        prior_dicts_list
    )

def main(epochs_finetune=8, quick_mode=False):
    start_pipeline_time = time.time()
    
    print("\n" + "=" * 80)
    print("STARTING HYBRID ML+DL VGG16-SVM BLOOD GROUP PREDICTION TRAINING PIPELINE")
    print("=" * 80)
    
    # 1. Dataset Discovery & Stratified Split
    records = load_dataset_records(max_samples_per_class=350 if not quick_mode else 100)
    train_paths, y_train, val_paths, y_val, test_paths, y_test = perform_stratified_70_15_15_split(records)
    
    # 2. VGG16 Preprocessing & Fine-Tuning
    print("\n[*] Preprocessing Train & Val Image Tensors for VGG16 Fine-Tuning...")
    X_train_tensors = np.array([apply_training_augmentation(preprocess_image_for_vgg16(p)) for p in train_paths], dtype=np.float32)
    X_val_tensors = np.array([preprocess_image_for_vgg16(p) for p in val_paths], dtype=np.float32)
    
    vgg_model, feature_extractor, history = train_and_finetune_vgg16(
        X_train_tensors, y_train,
        X_val_tensors, y_val,
        epochs=epochs_finetune,
        batch_size=32
    )
    
    # Save fine-tuned Keras model
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    try:
        vgg_model.save(VGG16_KERAS_SAVE_PATH)
        print(f"[OK] Saved fine-tuned VGG16 Keras backbone to: {VGG16_KERAS_SAVE_PATH}")
    except Exception as e:
        print(f"[!] Note on saving Keras model: {e}")
        
    # 3. Extract Multi-Modal Features
    print("\n[*] Extracting Multi-Modal Deep & Dermatoglyphic Features across Splits...")
    X_deep_train, X_ridge_train, priors_train = extract_features_for_split(train_paths, feature_extractor, augment=False)
    X_deep_val, X_ridge_val, priors_val = extract_features_for_split(val_paths, feature_extractor, augment=False)
    X_deep_test, X_ridge_test, priors_test = extract_features_for_split(test_paths, feature_extractor, augment=False)
    
    # Form Concatenated Vectors: [VGG16 Deep 512-dim + Ridge Biometrics 15-dim]
    X_hybrid_train = np.hstack([X_deep_train, X_ridge_train])
    X_hybrid_val = np.hstack([X_deep_val, X_ridge_val])
    X_hybrid_test = np.hstack([X_deep_test, X_ridge_test])
    
    print(f"  -> Extracted Deep Dimensions:   {X_deep_train.shape[1]}-dim")
    print(f"  -> Extracted Ridge Dimensions:  {X_ridge_train.shape[1]}-dim")
    print(f"  -> Combined Hybrid Feature Dim: {X_hybrid_train.shape[1]}-dim")
    
    # 4. Feature Standardization & PCA Dimensionality Reduction
    print("\n[*] Applying Standardization & PCA (Retaining 95% Cumulative Variance)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_hybrid_train)
    X_val_scaled = scaler.transform(X_hybrid_val)
    X_test_scaled = scaler.transform(X_hybrid_test)
    
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"  -> PCA Reduced Dimensions: {X_hybrid_train.shape[1]} -> {X_train_pca.shape[1]} components (95% variance)")
    
    # 5. 5-Fold Stratified Cross-Validation on Training Set
    print("\n" + "=" * 75)
    print("5-FOLD STRATIFIED CROSS-VALIDATION ON TRAINING SET")
    print("=" * 75)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_k_idx, val_k_idx) in enumerate(skf.split(X_train_pca, y_train), start=1):
        X_k_train, y_k_train = X_train_pca[train_k_idx], y_train[train_k_idx]
        X_k_val, y_k_val = X_train_pca[val_k_idx], y_train[val_k_idx]
        
        clf_fold = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42, class_weight='balanced')
        clf_fold.fit(X_k_train, y_k_train)
        score = clf_fold.score(X_k_val, y_k_val)
        cv_scores.append(score)
        print(f"  Fold {fold}: Accuracy = {score * 100:.2f}%")
        
    print(f"[*] 5-Fold Cross-Validation Mean Accuracy: {np.mean(cv_scores) * 100:.2f}% (+/- {np.std(cv_scores) * 100:.2f}%)")
    
    # 6. Hyperparameter GridSearch
    print("\n[*] Performing GridSearchCV over SVM Hyperparameters (Kernel, C, Gamma)...")
    param_grid = {
        'C': [1.0, 10.0, 50.0],
        'gamma': ['scale', 'auto', 0.01],
        'kernel': ['rbf', 'poly']
    }
    
    grid_search = GridSearchCV(
        SVC(probability=True, random_state=42, class_weight='balanced'),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train_pca, y_train)
    best_svm = grid_search.best_estimator_
    print(f"[OK] Best SVM Hyperparameters: {grid_search.best_params_} (CV Score: {grid_search.best_score_*100:.2f}%)")
    
    # 7. Fit Population Prevalence Calibrator
    calibrator = BayesianPrevalenceCalibrator(
        true_prevalence=REAL_WORLD_PREVALENCE,
        alpha_prevalence=0.35,
        beta_pattern=0.25
    )
    calibrator.fit_training_distribution(y_train, classes=CLASSES)
    
    # 8. Model Evaluations on Held-Out 15% Test Set
    print("\n" + "=" * 80)
    print("HELD-OUT 15% TEST SET EVALUATION & ABLATION STUDY")
    print("=" * 80)
    
    # Raw SVM Probabilities
    raw_test_probs = best_svm.predict_proba(X_test_pca)
    y_pred_baseline = np.argmax(raw_test_probs, axis=1)
    
    # Ablation 1: Baseline CNN+SVM
    acc_baseline = accuracy_score(y_test, y_pred_baseline) * 100.0
    f1_baseline = f1_score(y_test, y_pred_baseline, average='weighted') * 100.0
    
    # Ablation 2: + Pattern Priors
    probs_with_patterns = calibrator.calibrate_probabilities(raw_test_probs, pattern_priors_list=priors_test)
    # pattern-only shift (alpha=0, beta=0.25)
    calibrator_pattern_only = BayesianPrevalenceCalibrator(alpha_prevalence=0.0, beta_pattern=0.25)
    calibrator_pattern_only.fit_training_distribution(y_train, classes=CLASSES)
    probs_pattern_only = calibrator_pattern_only.calibrate_probabilities(raw_test_probs, pattern_priors_list=priors_test)
    y_pred_pattern = np.argmax(probs_pattern_only, axis=1)
    acc_pattern = accuracy_score(y_test, y_pred_pattern) * 100.0
    f1_pattern = f1_score(y_test, y_pred_pattern, average='weighted') * 100.0
    
    # Ablation 3: + Prevalence Calibration Only
    calibrator_prev_only = BayesianPrevalenceCalibrator(alpha_prevalence=0.35, beta_pattern=0.0)
    calibrator_prev_only.fit_training_distribution(y_train, classes=CLASSES)
    probs_prev_only = calibrator_prev_only.calibrate_probabilities(raw_test_probs, pattern_priors_list=None)
    y_pred_prev = np.argmax(probs_prev_only, axis=1)
    acc_prev = accuracy_score(y_test, y_pred_prev) * 100.0
    f1_prev = f1_score(y_test, y_pred_prev, average='weighted') * 100.0
    
    # Ablation 4: Full Hybrid (Both Combined)
    probs_full_hybrid = calibrator.calibrate_probabilities(raw_test_probs, pattern_priors_list=priors_test)
    y_pred_full = np.argmax(probs_full_hybrid, axis=1)
    acc_full = accuracy_score(y_test, y_pred_full) * 100.0
    f1_full = f1_score(y_test, y_pred_full, average='weighted') * 100.0
    
    # Print Ablation Summary Table
    print("\n--- ABLATION STUDY COMPARISON ON TEST SET ---")
    print(f"{'Configuration':<45} | {'Accuracy':<10} | {'F1-Score (Weighted)':<20}")
    print("-" * 80)
    print(f"{'1. Baseline (Fine-Tuned VGG16 + SVM)':<45} | {acc_baseline:6.2f}%    | {f1_baseline:6.2f}%")
    print(f"{'2. + Ridge-Pattern Domain Priors':<45} | {acc_pattern:6.2f}%    | {f1_pattern:6.2f}%")
    print(f"{'3. + Bayesian Population Prevalence Calibration':<45} | {acc_prev:6.2f}%    | {f1_prev:6.2f}%")
    print(f"{'4. Full Hybrid (VGG16 + Priors + Prevalence)':<45} | {acc_full:6.2f}%    | {f1_full:6.2f}%")
    print("-" * 80)
    
    # Per-Class Classification Report (Full Hybrid)
    print("\n--- FULL HYBRID CLASSIFICATION REPORT (PER-CLASS PRECISION / RECALL / F1) ---")
    target_names = [CLASSES[i] for i in sorted(list(set(y_test)))]
    print(classification_report(y_test, y_pred_full, target_names=target_names, digits=4))
    
    # Full Confusion Matrix
    print("--- FULL CONFUSION MATRIX ---")
    cm = confusion_matrix(y_test, y_pred_full)
    header = "      " + " ".join([f"{c:>5}" for c in target_names])
    print(header)
    for idx, row in enumerate(cm):
        print(f"{target_names[idx]:>5} " + " ".join([f"{val:>5d}" for val in row]))
        
    # Real-World Prevalence Calibration Alignment Check
    print("\n--- REAL-WORLD POPULATION PREVALENCE CALIBRATION CHECK ---")
    alignment, tv_dist = compare_distribution_alignment(y_pred_full, true_prevalence=REAL_WORLD_PREVALENCE, classes=CLASSES)
    print(f"{'Blood Group':<12} | {'Observed Test %':<16} | {'Expected Real %':<16} | {'Deviation %':<12}")
    print("-" * 65)
    for row in alignment:
        print(f"{row['class']:<12} | {row['observed_pct']:>13.2f}%  | {row['expected_prevalence_pct']:>13.2f}%  | {row['deviation_pct']:>+10.2f}%")
    print(f"\nTotal Variation (TV) Distance: {tv_dist}%\n")
    
    # 9. Permutation Feature Importance Analysis
    print("--- PERMUTATION FEATURE IMPORTANCE (PCA & BIOMETRIC MODALITIES) ---")
    try:
        perm_res = permutation_importance(best_svm, X_test_pca, y_test, n_repeats=5, random_state=42)
        top_indices = np.argsort(perm_res.importances_mean)[::-1][:8]
        print("Top Predictive PCA Component Dimensions:")
        for rank, idx in enumerate(top_indices, start=1):
            print(f"  Rank {rank}: Component #{idx+1:2d} -> Mean Importance: {perm_res.importances_mean[idx]:.4f} (+/- {perm_res.importances_std[idx]:.4f})")
    except Exception as e:
        print(f"  [!] Feature importance computation note: {e}")
        
    # 10. Save Complete Pipeline Artifact
    pipeline_artifact = {
        'model_name': 'Hybrid Fine-Tuned VGG16 + Dermatoglyphic SVM',
        'svm': best_svm,
        'scaler': scaler,
        'pca': pca,
        'classes': CLASSES,
        'calibrator': calibrator,
        'prior_multipliers': DEFAULT_PRIOR_MULTIPLIERS,
        'metrics': {
            'cv_mean_acc': round(float(np.mean(cv_scores) * 100), 2),
            'cv_std_acc': round(float(np.std(cv_scores) * 100), 2),
            'test_acc_baseline': round(acc_baseline, 2),
            'test_acc_pattern_priors': round(acc_pattern, 2),
            'test_acc_prevalence_calibrated': round(acc_prev, 2),
            'test_acc_full_hybrid': round(acc_full, 2),
            'test_f1_full_hybrid': round(f1_full, 2),
            'tv_distance': tv_dist
        },
        'vgg16_keras_model_path': VGG16_KERAS_SAVE_PATH
    }
    
    with open(HYBRID_VGG16_SAVE_PATH, 'wb') as f:
        pickle.dump(pipeline_artifact, f)
        
    print(f"\n[SUCCESS] Saved Complete Hybrid VGG16-SVM Pipeline Artifact to: {HYBRID_VGG16_SAVE_PATH}")
    print(f"Total Pipeline Execution Time: {time.time() - start_pipeline_time:.2f} seconds.")
    print("=" * 80)

if __name__ == '__main__':
    # Run pipeline with 8 fine-tuning epochs
    main(epochs_finetune=8)
