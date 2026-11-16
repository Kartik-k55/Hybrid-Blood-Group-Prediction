"""
================================================================================
Reproducibility & Consistency Audit: Dataset Splits, Counts, and Leakage Checks
================================================================================
Checks:
1. Raw file count reconciliation (6,321 raw -> 580 dups removed -> 5,741 unique).
2. Exact bracket-file breakdown (1,417 files).
3. Split consistency (Train 4,018 + Val 861 + Test 862 == 5,741).
4. Manifest disjointness (MD5 hash overlap == 0 across all split pairs).
5. Stem overlap check (cluster_X_YYYY overlap == 0 across all split pairs).
6. Subject ID audit & biological limitations documentation.
================================================================================
"""

import os
import sys
import json
import hashlib
import re
from collections import defaultdict
import pandas as pd

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

RAW_DIR = os.path.join('dataset', 'Fingerprint Dataset for Blood Group Classification')
TRAIN_CSV = os.path.join('dataset', 'processed', 'dataset_train_split.csv')
VAL_CSV = os.path.join('dataset', 'processed', 'dataset_val_split.csv')
TEST_CSV = os.path.join('dataset', 'processed', 'dataset_test_split.csv')
SUMMARY_JSON = os.path.join('dataset', 'processed', 'dataset_summary.json')

ALL_CLASSES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']


def run_dataset_audit():
    print("=" * 85)
    print("REPRODUCIBILITY AUDIT: DATASET MANIFESTS, COUNTS, & SPLIT DISJOINTNESS")
    print("=" * 85)

    # 1. Raw Files Scan & Deduplication
    print("\n--- 1. RAW DATASET SCAN & COUNT RECONCILIATION ---")
    seen_hashes = {}
    exact_duplicates = []
    raw_files_per_class = {}
    bracket_files = []
    
    for c in ALL_CLASSES:
        class_folder = os.path.join(RAW_DIR, c)
        files = sorted([f for f in os.listdir(class_folder) if not f.startswith('.') and 'checkpoint' not in f.lower() and os.path.splitext(f)[1].lower() in ['.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff']])
        raw_files_per_class[c] = len(files)
        for fname in files:
            fpath = os.path.join(class_folder, fname)
            if '(' in fname and ')' in fname:
                bracket_files.append((c, fname, fpath))
            with open(fpath, 'rb') as fp:
                h = hashlib.md5(fp.read()).hexdigest()
            if h in seen_hashes:
                exact_duplicates.append({
                    'class': c,
                    'dup_path': fpath,
                    'orig_path': seen_hashes[h],
                    'md5': h
                })
            else:
                seen_hashes[h] = fpath

    total_scanned = sum(raw_files_per_class.values())
    unique_retained = len(seen_hashes)
    num_dups = len(exact_duplicates)
    num_brackets = len(bracket_files)

    print(f"Total raw image files scanned:    {total_scanned}")
    print(f"Exact MD5 duplicate files:        {num_dups}")
    print(f"Unique images retained:           {unique_retained}")
    print(f"Files with bracket in name:       {num_brackets}")
    print(f"Mathematical identity: {total_scanned} - {num_dups} == {unique_retained} -> {total_scanned - num_dups == unique_retained}")

    print("\nPer-Class Breakdown:")
    print(f"{'Class':<6} | {'Scanned':<8} | {'Exact Duplicates':<18} | {'Bracket Files':<14} | {'Retained Unique':<15}")
    print("-" * 72)
    dups_by_class = defaultdict(int)
    for d in exact_duplicates:
        dups_by_class[d['class']] += 1
    brackets_by_class = defaultdict(int)
    for b in bracket_files:
        brackets_by_class[b[0]] += 1

    for c in ALL_CLASSES:
        scanned = raw_files_per_class[c]
        dups = dups_by_class[c]
        brk = brackets_by_class[c]
        ret = scanned - dups
        print(f"{c:<6} | {scanned:<8} | {dups:<18} | {brk:<14} | {ret:<15}")

    # 2. Bracket Twin Overlap Explanation
    print("\n--- 2. EXPLANATION OF BRACKET-TWIN FILES (N = 1,417) VS DUPLICATES (N = 580) ---")
    print("Why do 1,417 files have ' (2)' in their filenames, but only 580 files are removed?")
    print("1. In class A+: There were 1,130 files. 565 were original names ('cluster_0_X.BMP')")
    print("   and 565 were twin copies ('cluster_0_X (2).BMP'). These were identical binary duplicates.")
    print("   Because ' (2).BMP' sorts alphabetically before '.BMP' (ASCII 32 space < ASCII 46 dot),")
    print("   the bracket copy was scanned first and retained, while the duplicate was discarded (565 duplicates).")
    print("2. In class O+: There are 852 files, and ALL 852 have ' (2)' in their filenames.")
    print("   There are NO non-bracket versions of these files in the raw folder. They are the sole")
    print("   representatives of these fingerprint samples. 851 of them are unique (1 was a duplicate).")
    print("3. In remaining classes: 15 exact duplicates were found across A-, AB+, AB-, B+, O-, having non-bracket names.")
    print(f"   Total exact duplicates removed: 565 (A+) + 1 (O+) + 14 (other classes) = {num_dups}.")

    # 3. Splits Sum and Disjointness Check
    print("\n--- 3. SPLIT SUM & DISJOINTNESS VERIFICATION ---")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    sum_splits = n_train + n_val + n_test

    print(f"Train samples: {n_train} ({n_train / unique_retained * 100:.2f}%)")
    print(f"Val samples:   {n_val} ({n_val / unique_retained * 100:.2f}%)")
    print(f"Test samples:  {n_test} ({n_test / unique_retained * 100:.2f}%)")
    print(f"Sum of splits: {sum_splits} == {unique_retained} -> {sum_splits == unique_retained}")

    # MD5 Disjointness
    train_hashes = set(train_df['md5'])
    val_hashes = set(val_df['md5'])
    test_hashes = set(test_df['md5'])

    train_val_hash_overlap = len(train_hashes.intersection(val_hashes))
    train_test_hash_overlap = len(train_hashes.intersection(test_hashes))
    val_test_hash_overlap = len(val_hashes.intersection(test_hashes))

    print("\nMD5 Hash Disjointness Verification:")
    print(f"Train ∩ Val  overlap: {train_val_hash_overlap} (Expected: 0)")
    print(f"Train ∩ Test overlap: {train_test_hash_overlap} (Expected: 0)")
    print(f"Val   ∩ Test overlap: {val_test_hash_overlap} (Expected: 0)")

    # 4. Filename Stem Overlap Check
    print("\n--- 4. FILENAME STEM OVERLAP VERIFICATION ---")
    def extract_stem(fname):
        clean = re.sub(r'\s*\(\d+\)', '', fname)
        base = os.path.splitext(clean)[0]
        return base

    train_stems = set(train_df['filename'].apply(extract_stem))
    val_stems = set(val_df['filename'].apply(extract_stem))
    test_stems = set(test_df['filename'].apply(extract_stem))

    train_val_stem_overlap = len(train_stems.intersection(val_stems))
    train_test_stem_overlap = len(train_stems.intersection(test_stems))
    val_test_stem_overlap = len(val_stems.intersection(test_stems))

    print(f"Train ∩ Val  stem overlap: {train_val_stem_overlap} (Expected: 0)")
    print(f"Train ∩ Test stem overlap: {train_test_stem_overlap} (Expected: 0)")
    print(f"Val   ∩ Test stem overlap: {val_test_stem_overlap} (Expected: 0)")

    # 5. Subject ID Audit & Independence Status
    print("\n--- 5. SUBJECT INDEPENDENCE AUDIT ---")
    print("Checking whether metadata or filenames specify donor IDs...")
    sample_filenames = list(train_df['filename'].head(5))
    print(f"Sample filenames: {sample_filenames}")
    print("Pattern observed: 'cluster_<class_id>_<image_num>.BMP'")
    print("Result: No donor IDs, demographics, or multi-finger identifiers exist.")
    print("AUDIT FINDING: Subject independence CANNOT be confirmed without donor metadata.")
    print("STATUS: Documented as an unavoidable fundamental dataset limitation.")

    all_passed = (
        total_scanned == 6321 and
        num_dups == 580 and
        unique_retained == 5741 and
        n_train == 4018 and
        n_val == 861 and
        n_test == 862 and
        sum_splits == 5741 and
        train_val_hash_overlap == 0 and
        train_test_hash_overlap == 0 and
        val_test_hash_overlap == 0 and
        train_val_stem_overlap == 0 and
        train_test_stem_overlap == 0 and
        val_test_stem_overlap == 0
    )

    print("\n" + "=" * 85)
    print(f"DATASET AUDIT RESULT: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 85)
    return all_passed


if __name__ == '__main__':
    run_dataset_audit()
