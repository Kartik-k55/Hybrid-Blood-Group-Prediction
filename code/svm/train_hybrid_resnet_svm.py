import os
import sys
import glob
import time
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Ensure UTF-8 output encoding for windows consoles
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATASET_DIR = os.path.join('dataset', 'dataset_blood_group')
RESNET_MODEL_PATH = os.path.join('test', 'model_blood_group_detection_resnet.h5')
HYBRID_MODEL_SAVE_PATH = os.path.join('test', 'model_blood_group_detection_hybrid_resnet_svm.pkl')
SVM_MODEL_SAVE_PATH = os.path.join('test', 'model_blood_group_detection_svm.pkl')

CLASSES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Descriptions of learned biometric and morphological patterns per blood group
BLOOD_GROUP_PATTERNS = {
    'A+': 'Ridge density, ridge spacing, orientation, texture, minutiae distribution and overall pattern learned from A+ samples',
    'A-': 'Ridge density, spacing, ridge orientation, texture and minutiae distribution learned from A- samples',
    'B+': 'Ridge structure, density, orientation, texture and minutiae patterns learned from B+ samples',
    'B-': 'Ridge density, spacing, orientation, texture and local ridge characteristics learned from B- samples',
    'AB+': 'Combination of ridge, texture, orientation and minutiae characteristics learned from AB+ samples',
    'AB-': 'Ridge density, spacing, orientation, texture and minutiae characteristics learned from AB- samples',
    'O+': 'Ridge density, spacing, orientation, texture, minutiae and global pattern characteristics learned from O+ samples',
    'O-': 'Ridge density, spacing, orientation, texture and minutiae characteristics learned from O- samples'
}

def extract_advanced_ridge_biometrics(img):
    """
    Extract comprehensive biometric, ridge structure, orientation, texture,
    spacing, and minutiae distribution descriptors from a fingerprint image.
    """
    gray = img.convert('L')
    w, h = gray.size
    arr = np.array(gray, dtype=np.float32)
    
    # 1. Ridge Density & Mean Intensity
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    gx, gy = np.gradient(arr)
    mag = np.hypot(gx, gy)
    ridge_density = float(np.mean(mag))
    
    # 2. Ridge Spacing (Peak-to-Peak frequency analysis)
    # Sample horizontal and vertical profiles across the center
    center_y, center_x = h // 2, w // 2
    row_profile = arr[center_y, :]
    col_profile = arr[:, center_x]
    
    # Compute zero-crossings / peak frequencies as proxy for ridge spacing
    row_diff = np.diff(np.sign(np.diff(row_profile)))
    col_diff = np.diff(np.sign(np.diff(col_profile)))
    row_peaks = np.where(row_diff < 0)[0]
    col_peaks = np.where(col_diff < 0)[0]
    
    row_spacing = float(np.mean(np.diff(row_peaks))) if len(row_peaks) > 1 else 10.0
    col_spacing = float(np.mean(np.diff(col_peaks))) if len(col_peaks) > 1 else 10.0
    mean_ridge_spacing = float((row_spacing + col_spacing) / 2.0)
    
    # 3. Ridge Orientation & Coherence Field
    gx2 = gx * gx
    gy2 = gy * gy
    gxy = gx * gy
    
    gxx_m_gyy = np.mean(gx2 - gy2)
    two_gxy = np.mean(2.0 * gxy)
    orientation_angle = 0.5 * np.arctan2(two_gxy, gxx_m_gyy) * (180.0 / np.pi) % 180.0
    
    coherence_numerator = np.hypot(np.mean(gx2 - gy2), np.mean(2.0 * gxy))
    coherence_denominator = np.mean(gx2 + gy2) + 1e-5
    orientation_coherence = float(coherence_numerator / coherence_denominator)
    
    # 4. Minutiae Distribution (Corner / Bifurcation / Termination proxy across 5 spatial zones)
    # Threshold gradients to identify high curvature minutiae points
    minutiae_map = (mag > (mean_val + 0.5 * std_val)).astype(np.float32)
    
    tl = float(np.mean(minutiae_map[:h//2, :w//2]))
    tr = float(np.mean(minutiae_map[:h//2, w//2:]))
    bl = float(np.mean(minutiae_map[h//2:, :w//2]))
    br = float(np.mean(minutiae_map[h//2:, w//2:]))
    center_zone = float(np.mean(minutiae_map[h//4:3*h//4, w//4:3*w//4]))
    minutiae_density = float(np.mean(minutiae_map))
    
    # 5. Texture & Local Pattern Descriptors (Multi-scale HOG & moments)
    img_res = gray.resize((64, 64), Image.Resampling.BILINEAR)
    res_arr = np.array(img_res, dtype=np.float32)
    rx = np.zeros_like(res_arr)
    ry = np.zeros_like(res_arr)
    rx[:, 1:-1] = res_arr[:, 2:] - res_arr[:, :-2]
    ry[1:-1, :] = res_arr[2:, :] - res_arr[:-2, :]
    
    rmag = np.hypot(rx, ry)
    rori = np.arctan2(ry, rx) * (180.0 / np.pi) % 180.0
    
    cell_size = 16
    num_bins = 8
    bin_width = 180.0 / num_bins
    hog_feats = []
    
    for r in range(0, 64, cell_size):
        for c in range(0, 64, cell_size):
            mag_c = rmag[r:r+cell_size, c:c+cell_size]
            ori_c = rori[r:r+cell_size, c:c+cell_size]
            hist = np.zeros(num_bins, dtype=np.float32)
            for b in range(num_bins):
                mask = (ori_c >= b * bin_width) & (ori_c < (b + 1) * bin_width)
                hist[b] = np.sum(mag_c[mask])
            norm = np.linalg.norm(hist) + 1e-5
            hog_feats.extend(hist / norm)
            
    # Assembled Biometric Feature Vector
    bio_stats = np.array([
        mean_val,
        std_val,
        ridge_density,
        mean_ridge_spacing,
        orientation_angle,
        orientation_coherence,
        minutiae_density,
        tl, tr, bl, br, center_zone
    ], dtype=np.float32)
    
    feature_vector = np.hstack([bio_stats, np.array(hog_feats, dtype=np.float32)])
    
    readable_biometrics = {
        'dimensions': f"{w}x{h}",
        'mean_intensity': round(mean_val, 2),
        'contrast_std': round(std_val, 2),
        'ridge_density_score': round(ridge_density, 2),
        'ridge_spacing_px': round(mean_ridge_spacing, 2),
        'dominant_orientation_deg': round(orientation_angle, 1),
        'orientation_coherence': round(orientation_coherence, 3),
        'minutiae_density_score': round(minutiae_density, 3),
        'spatial_minutiae_zones': {
            'top_left': round(tl, 3),
            'top_right': round(tr, 3),
            'bottom_left': round(bl, 3),
            'bottom_right': round(br, 3),
            'center_core': round(center_zone, 3)
        }
    }
    
    return feature_vector, readable_biometrics

def preprocess_for_resnet(img):
    img = img.convert('RGB')
    img = img.resize((256, 256), Image.Resampling.BILINEAR)
    x = np.array(img, dtype=np.float32)
    try:
        from tensorflow.keras.applications.resnet50 import preprocess_input
        x = preprocess_input(x)
    except Exception:
        x = x[..., ::-1]
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68
    return x

def build_feature_extractor():
    print(f"Loading ResNet Deep Backbone from {RESNET_MODEL_PATH}...")
    resnet_full = tf.keras.models.load_model(RESNET_MODEL_PATH, compile=False)
    feature_extractor = tf.keras.Model(
        inputs=resnet_full.inputs, 
        outputs=resnet_full.layers[-2].output
    )
    print("ResNet 128-dimensional Feature Extractor constructed successfully.")
    return feature_extractor

def load_dataset_and_extract(feature_extractor, max_samples_per_class=300, batch_size=64):
    print(f"Extracting multi-modal biometric and deep patterns across all {len(CLASSES)} blood groups...")
    
    all_img_paths = []
    all_labels = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_folder = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_folder):
            print(f"Warning: Folder {class_folder} not found. Skipping.")
            continue
            
        file_list = glob.glob(os.path.join(class_folder, '*.BMP')) + \
                    glob.glob(os.path.join(class_folder, '*.bmp')) + \
                    glob.glob(os.path.join(class_folder, '*.jpg'))
                    
        np.random.seed(42)
        n_select = min(len(file_list), max_samples_per_class)
        selected_files = np.random.choice(file_list, n_select, replace=False)
        all_img_paths.extend(selected_files)
        all_labels.extend([class_idx] * len(selected_files))
        print(f"  [+] Class '{class_name}': selected {n_select} samples")
        
    print(f"Total dataset images selected for training: {len(all_img_paths)}")
    
    X_hybrid = []
    X_bio_only = []
    y_labels = []
    
    for i in range(0, len(all_img_paths), batch_size):
        batch_paths = all_img_paths[i:i+batch_size]
        batch_targets = all_labels[i:i+batch_size]
        
        batch_preprocessed = []
        batch_bios = []
        valid_indices = []
        
        for idx, fpath in enumerate(batch_paths):
            try:
                with Image.open(fpath) as img:
                    px = preprocess_for_resnet(img)
                    bio_vec, _ = extract_advanced_ridge_biometrics(img)
                    batch_preprocessed.append(px)
                    batch_bios.append(bio_vec)
                    valid_indices.append(idx)
            except Exception as e:
                print(f"Error reading image {fpath}: {e}")
                
        if not batch_preprocessed:
            continue
            
        batch_tensor = np.array(batch_preprocessed, dtype=np.float32)
        deep_embeddings = feature_extractor.predict(batch_tensor, batch_size=len(batch_tensor), verbose=0)
        
        for idx_in_batch in range(len(deep_embeddings)):
            deep_vec = deep_embeddings[idx_in_batch]
            bio_vec = batch_bios[idx_in_batch]
            hybrid_vec = np.hstack([deep_vec, bio_vec])
            
            X_hybrid.append(hybrid_vec)
            X_bio_only.append(bio_vec)
            y_labels.append(batch_targets[valid_indices[idx_in_batch]])
            
        print(f"  -> Extracted features for {min(i + batch_size, len(all_img_paths))}/{len(all_img_paths)} samples...")
        
    return np.array(X_hybrid, dtype=np.float32), np.array(X_bio_only, dtype=np.float32), np.array(y_labels, dtype=np.int64)

def main():
    start_time = time.time()
    feature_extractor = build_feature_extractor()
    
    # Extract features
    X_hybrid, X_bio, y = load_dataset_and_extract(feature_extractor, max_samples_per_class=350, batch_size=64)
    print(f"\nFeature Extraction Complete! Hybrid Matrix: {X_hybrid.shape} | Standalone Biometric Matrix: {X_bio.shape}")
    
    # Train / Test split (80-20)
    X_train_h, X_test_h, y_train, y_test = train_test_split(
        X_hybrid, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train_h.shape[0]} samples | Testing set: {X_test_h.shape[0]} samples")
    
    # Feature Standardization
    scaler_hybrid = StandardScaler()
    X_train_h_scaled = scaler_hybrid.fit_transform(X_train_h)
    X_test_h_scaled = scaler_hybrid.transform(X_test_h)
    
    # Train Hybrid ResNet-SVM Classifier
    print("\nTraining Hybrid ResNet-SVM Multi-Class Classifier (RBF Kernel, C=10.0, probability=True)...")
    svm_hybrid = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42, class_weight='balanced')
    svm_hybrid.fit(X_train_h_scaled, y_train)
    
    y_pred_train_h = svm_hybrid.predict(X_train_h_scaled)
    y_pred_test_h = svm_hybrid.predict(X_test_h_scaled)
    
    train_acc_h = accuracy_score(y_train, y_pred_train_h) * 100
    test_acc_h = accuracy_score(y_test, y_pred_test_h) * 100
    
    print("\n========================================================")
    print(f"HYBRID ResNet-SVM Training Accuracy: {train_acc_h:.2f}%")
    print(f"HYBRID ResNet-SVM Testing Accuracy:  {test_acc_h:.2f}%")
    print("========================================================\n")
    
    print("Classification Report on Test Set:")
    target_names = [CLASSES[i] for i in sorted(list(set(y_test)))]
    print(classification_report(y_test, y_pred_test_h, target_names=target_names))
    
    # Save Hybrid ResNet-SVM Pipeline Artifact
    os.makedirs(os.path.dirname(HYBRID_MODEL_SAVE_PATH), exist_ok=True)
    with open(HYBRID_MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump({
            'svm': svm_hybrid,
            'scaler': scaler_hybrid,
            'classes': CLASSES,
            'patterns': BLOOD_GROUP_PATTERNS,
            'train_acc': round(train_acc_h, 2),
            'test_acc': round(test_acc_h, 2),
            'feature_dim': X_hybrid.shape[1]
        }, f)
    print(f"[OK] Saved Hybrid ResNet-SVM model artifact to: {HYBRID_MODEL_SAVE_PATH}")
    
    # Also Train and Save Standalone Biometric SVM
    X_train_b, X_test_b, _, _ = train_test_split(X_bio, y, test_size=0.2, random_state=42, stratify=y)
    scaler_bio = StandardScaler()
    X_train_b_scaled = scaler_bio.fit_transform(X_train_b)
    X_test_b_scaled = scaler_bio.transform(X_test_b)
    
    print("\nTraining Standalone Biometric Ridge & Spacing SVM Classifier...")
    svm_bio = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42, class_weight='balanced')
    svm_bio.fit(X_train_b_scaled, y_train)
    
    with open(SVM_MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump({
            'svm': svm_bio,
            'scaler': scaler_bio,
            'classes': CLASSES,
            'patterns': BLOOD_GROUP_PATTERNS,
            'train_acc': round(accuracy_score(y_train, svm_bio.predict(X_train_b_scaled)) * 100, 2),
            'test_acc': round(accuracy_score(y_test, svm_bio.predict(X_test_b_scaled)) * 100, 2),
            'feature_dim': X_bio.shape[1]
        }, f)
    print(f"[OK] Saved Standalone Biometric SVM model artifact to: {SVM_MODEL_SAVE_PATH}")
    
    print(f"\nModel training and pattern optimization complete in {time.time() - start_time:.2f} seconds!")

if __name__ == '__main__':
    main()
