import os
import sys
import glob
import time
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Ensure UTF-8 output encoding for windows consoles
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

CLASSES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
DATASET_DIRS = [
    os.path.join('dataset', 'dataset_blood_group'),
    os.path.join('dataset', 'Fingerprint Dataset for Blood Group Classification')
]
OUTPUT_DIR = os.path.join('dataset', 'processed')

def step1_discover_and_validate(dataset_dirs):
    print("=" * 70)
    print("STEP 1: DATASET DISCOVERY & IMAGE INTEGRITY VALIDATION")
    print("=" * 70)
    
    selected_dir = None
    for d in dataset_dirs:
        if os.path.exists(d):
            selected_dir = d
            break
            
    if not selected_dir:
        raise FileNotFoundError(f"No valid dataset directory found in {dataset_dirs}")
        
    print(f"[*] Reading dataset from: {selected_dir}")
    
    records = []
    corrupted_count = 0
    
    for class_idx, class_name in enumerate(CLASSES):
        class_folder = os.path.join(selected_dir, class_name)
        if not os.path.exists(class_folder):
            print(f"  [!] Folder {class_folder} not found, skipping.")
            continue
            
        patterns = ['*.BMP', '*.bmp', '*.jpg', '*.jpeg', '*.png', '*.PNG']
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(class_folder, p)))
        files = sorted(list(set(files)))
        
        valid_class_files = 0
        for fpath in files:
            try:
                with Image.open(fpath) as img:
                    width, height = img.size
                    mode = img.mode
                    records.append({
                        'filepath': fpath.replace('\\', '/'),
                        'filename': os.path.basename(fpath),
                        'label': class_name,
                        'class_id': class_idx,
                        'width': width,
                        'height': height,
                        'mode': mode,
                        'file_size_bytes': os.path.getsize(fpath)
                    })
                    valid_class_files += 1
            except Exception as e:
                print(f"  [X] Corrupted file {fpath}: {e}")
                corrupted_count += 1
                
        print(f"  [+] Class '{class_name}': {valid_class_files} valid images verified.")

    df = pd.DataFrame(records)
    print(f"\n[OK] Step 1 Complete: Total {len(df)} valid images verified ({corrupted_count} corrupted).")
    return df, selected_dir

def step2_stratified_train_test_split(df, test_size=0.20, random_state=42):
    print("\n" + "=" * 70)
    print("STEP 2: STRATIFIED DATASET SPLIT (TRAIN / VALIDATION / TEST)")
    print("=" * 70)
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
    
    print(f"[*] Total Samples: {len(df)}")
    print(f"[*] Training Set ({(1-test_size)*100:.0f}%): {len(train_df)} samples")
    print(f"[*] Test / Evaluation Set ({test_size*100:.0f}%): {len(test_df)} samples")
    
    print("\n  Class Distribution in Splits:")
    dist = pd.DataFrame({
        'Train': train_df['label'].value_counts(),
        'Test': test_df['label'].value_counts(),
        'Total': df['label'].value_counts()
    }).loc[CLASSES]
    print(dist.to_string())
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def step3_extract_biometric_features(df, max_sample_features=None):
    print("\n" + "=" * 70)
    print("STEP 3: BIOMETRIC & MORPHOLOGICAL FEATURE EXTRACTION")
    print("=" * 70)
    print("[*] Computing ridge density, mean intensity, contrast, and morphological properties...")
    
    target_df = df if max_sample_features is None else df.head(max_sample_features)
    start_time = time.time()
    
    bio_records = []
    for idx, row in target_df.iterrows():
        try:
            with Image.open(row['filepath']) as img:
                gray = img.convert('L')
                arr = np.array(gray, dtype=np.float32)
                mean_intensity = float(np.mean(arr))
                std_contrast = float(np.std(arr))
                gx, gy = np.gradient(arr)
                ridge_density = float(np.mean(np.hypot(gx, gy)))
                
                bio_records.append({
                    'mean_intensity': round(mean_intensity, 2),
                    'std_contrast': round(std_contrast, 2),
                    'ridge_density': round(ridge_density, 2)
                })
        except Exception as e:
            bio_records.append({
                'mean_intensity': 0.0,
                'std_contrast': 0.0,
                'ridge_density': 0.0
            })
            
        if (idx + 1) % 1000 == 0 or (idx + 1) == len(target_df):
            print(f"  -> Processed {idx + 1}/{len(target_df)} fingerprints ({(idx+1)/len(target_df)*100:.1f}%)")
            
    bio_df = pd.DataFrame(bio_records)
    combined_df = pd.concat([target_df.reset_index(drop=True), bio_df], axis=1)
    
    elapsed = time.time() - start_time
    print(f"[OK] Step 3 Complete: Biometric features computed in {elapsed:.2f}s.")
    return combined_df

def step4_save_processed_dataset(full_df, train_df, test_df, output_dir):
    print("\n" + "=" * 70)
    print("STEP 4: SAVING PROCESSED DATASET ARTIFACTS & METADATA")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    full_meta_path = os.path.join(output_dir, 'dataset_metadata_full.csv')
    train_path = os.path.join(output_dir, 'dataset_train_split.csv')
    test_path = os.path.join(output_dir, 'dataset_test_split.csv')
    summary_path = os.path.join(output_dir, 'dataset_summary.json')
    
    full_df.to_csv(full_meta_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    summary = {
        'total_images': int(len(full_df)),
        'train_samples': int(len(train_df)),
        'test_samples': int(len(test_df)),
        'classes': CLASSES,
        'class_counts': {k: int(v) for k, v in full_df['label'].value_counts().to_dict().items()},
        'mean_ridge_density_by_class': {k: round(float(v), 2) for k, v in full_df.groupby('label')['ridge_density'].mean().to_dict().items()},
        'mean_intensity_by_class': {k: round(float(v), 2) for k, v in full_df.groupby('label')['mean_intensity'].mean().to_dict().items()},
        'mean_contrast_by_class': {k: round(float(v), 2) for k, v in full_df.groupby('label')['std_contrast'].mean().to_dict().items()}
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
        
    print(f"  [OK] Full metadata saved to: {full_meta_path}")
    print(f"  [OK] Train split saved to: {train_path}")
    print(f"  [OK] Test split saved to: {test_path}")
    print(f"  [OK] Summary JSON saved to: {summary_path}")
    return summary

def main():
    start_all = time.time()
    print("\n" + "#" * 70)
    print("   RAKTA MITRA: FINGERPRINT BLOOD GROUP DATASET PIPELINE (FROM STEP 1)")
    print("#" * 70 + "\n")
    
    # Step 1: Discover and Validate Raw Images
    raw_df, dataset_path = step1_discover_and_validate(DATASET_DIRS)
    
    # Step 2: Biometric Feature Extraction
    processed_df = step3_extract_biometric_features(raw_df)
    
    # Step 3: Stratified Split with Features
    train_processed, test_processed = step2_stratified_train_test_split(processed_df, test_size=0.20)
    
    # Step 4: Save processed files and summaries
    summary = step4_save_processed_dataset(processed_df, train_processed, test_processed, OUTPUT_DIR)
    
    total_time = time.time() - start_all
    print("\n" + "=" * 70)
    print(f"DATASET PROCESSING COMPLETE IN {total_time:.2f} SECONDS!")
    print("=" * 70)

if __name__ == '__main__':
    main()
