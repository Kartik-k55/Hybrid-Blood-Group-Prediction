import os
import glob
import time
import pickle
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset configuration
DATASET_DIR = os.path.join('dataset', 'dataset_blood_group')
MODEL_SAVE_PATH = os.path.join('test', 'model_blood_group_detection_svm.joblib')
CLASSES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

def compute_simple_hog(gray_img_arr):
    """
    Compute cell-based Histogram of Oriented Gradients (HOG) features.
    """
    # Downsample image for spatial gradient calculation
    img_res = Image.fromarray(gray_img_arr).resize((64, 64), Image.Resampling.BILINEAR)
    arr = np.array(img_res, dtype=np.float32)
    
    # Compute spatial gradients
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
    gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
    
    magnitude = np.hypot(gx, gy)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
    
    # Grid 8x8 cells -> 8x8 grid of cells across 64x64 image
    cell_size = 8
    num_bins = 8
    bin_width = 180 / num_bins
    hog_feats = []
    
    for r in range(0, 64, cell_size):
        for c in range(0, 64, cell_size):
            mag_cell = magnitude[r:r+cell_size, c:c+cell_size]
            ori_cell = orientation[r:r+cell_size, c:c+cell_size]
            hist = np.zeros(num_bins, dtype=np.float32)
            
            for b in range(num_bins):
                mask = (ori_cell >= b * bin_width) & (ori_cell < (b + 1) * bin_width)
                hist[b] = np.sum(mag_cell[mask])
                
            norm = np.linalg.norm(hist) + 1e-5
            hog_feats.extend(hist / norm)
            
    return np.array(hog_feats, dtype=np.float32)

def extract_image_features(img_path_or_pil):
    """
    Extract comprehensive statistical and spatial ridge features for SVM classification.
    """
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil)
    else:
        img = img_path_or_pil
        
    gray = img.convert('L')
    arr = np.array(gray, dtype=np.float32)
    
    # 1. Global Intensity & Texture Statistics
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    p25, p50, p75 = np.percentile(arr, [25, 50, 75])
    
    # 2. Downsampled Intensity Vector
    thumb = gray.resize((32, 32), Image.Resampling.BILINEAR)
    thumb_vec = (np.array(thumb, dtype=np.float32).flatten() - mean_val) / (std_val + 1e-5)
    
    # 3. HOG Features
    hog_vec = compute_simple_hog(arr)
    
    # Concatenate feature vector
    stats_vec = np.array([mean_val, std_val, min_val, max_val, p25, p50, p75], dtype=np.float32)
    feature_vector = np.hstack([stats_vec, thumb_vec, hog_vec])
    return feature_vector

def load_dataset(max_samples_per_class=200):
    print("Loading fingerprint dataset for SVM training...")
    X, y = [], []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_folder = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_folder):
            print(f"Warning: Directory {class_folder} not found. Skipping.")
            continue
            
        file_list = glob.glob(os.path.join(class_folder, '*.BMP')) + \
                    glob.glob(os.path.join(class_folder, '*.jpg')) + \
                    glob.glob(os.path.join(class_folder, '*.png'))
                    
        print(f"Class '{class_name}': Found {len(file_list)} images. Sampling up to {max_samples_per_class}...")
        np.random.seed(42)
        selected_files = np.random.choice(file_list, min(len(file_list), max_samples_per_class), replace=False)
        
        for fpath in selected_files:
            try:
                feat = extract_image_features(fpath)
                X.append(feat)
                y.append(class_idx)
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
                
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def main():
    start_time = time.time()
    X, y = load_dataset(max_samples_per_class=150)
    print(f"Extracted dataset features. Total samples: {X.shape[0]}, Feature dimension: {X.shape[1]}")
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM Model
    print("Training Support Vector Classifier (RBF Kernel, C=10.0, probability=True)...")
    svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred_train = svm_model.predict(X_train_scaled)
    y_pred_test = svm_model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_pred_train) * 100
    test_acc = accuracy_score(y_test, y_pred_test) * 100
    
    print("\n==========================================")
    print(f"SVM Model Training Accuracy: {train_acc:.2f}%")
    print(f"SVM Model Testing Accuracy:  {test_acc:.2f}%")
    print("==========================================\n")
    print("Classification Report on Test Data:")
    target_names = [CLASSES[i] for i in sorted(list(set(y_test)))]
    print(classification_report(y_test, y_pred_test, target_names=target_names))
    
    # Save Pipeline
    save_pkl_path = MODEL_SAVE_PATH.replace('.joblib', '.pkl')
    os.makedirs(os.path.dirname(save_pkl_path), exist_ok=True)
    with open(save_pkl_path, 'wb') as f:
        pickle.dump({
            'svm': svm_model,
            'scaler': scaler,
            'classes': CLASSES,
            'train_acc': round(train_acc, 2),
            'test_acc': round(test_acc, 2)
        }, f)
    
    print(f"Saved SVM model artifact successfully to: {save_pkl_path}")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
