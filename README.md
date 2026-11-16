# Blood Group Prediction from Fingerprint Images: Hybrid Deep Learning & Machine Learning

An audited, optimized, and statistically validated research project evaluating hybrid Deep Learning (CNN) and Classical Machine Learning (SVM) approaches to predict ABO/Rh blood groups from fingerprint impressions.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Audited Dataset & Splitting Methodology](#audited-dataset--splitting-methodology)
- [Architectures & Feature Pipelines](#architectures--feature-pipelines)
- [Verified Benchmark Results (Held-Out Test Set)](#verified-benchmark-results-held-out-test-set)
- [Supplementary Balanced Subset Analysis](#supplementary-balanced-subset-analysis)
- [Scientific & Clinical Limitations](#scientific--clinical-limitations)
- [Cryptographic Security vs. Medical Reality](#cryptographic-security-vs-medical-reality)
- [USB Fingerprint Scanner Integration & Hardware Guide](#usb-fingerprint-scanner-integration--hardware-guide)
- [Reproducible Setup & Execution Commands](#reproducible-setup--execution-commands)

---

## Project Overview

This repository investigates the classification of eight human blood groups (**A+, A-, AB+, AB-, B+, B-, O+, O-**) from digital fingerprint impressions using:
- **Deep Convolutional Neural Networks:** ResNet-50 and Fine-Tuned VGG16
- **Biometric Feature Engineering:** Peak-to-peak ridge spacing, orientation field coherence, and spatial minutiae density
- **Classical Descriptors:** Multi-scale Histogram of Oriented Gradients (HOG)
- **Classifiers & Calibration:** Support Vector Machines (RBF/Linear kernels with Platt scaling), PCA dimensionality reduction, and Bayesian population prevalence adjustments
- **Multi-Model Consensus:** Calibrated weighted ensemble combining deep convolutional features and morphological dermatoglyphic priors
- **Deployment Interface:** A local Flask web application with AES-256-GCM authenticated encryption for biometric records

---

## Audited Dataset & Splitting Methodology

### Forensic Audit Findings
1. **Deduplication & Count Reconciliation:**
   - **6,321** raw image files scanned across 8 class directories.
   - **580** exact MD5 duplicates detected and removed ($6,321 - 580 = 5,741$).
   - **1,417** files contain copy brackets (` (2)` in filename): In class `A+`, 565 were twin duplicate copies of original impressions and removed; in class `O+`, all 852 images possess bracket stems as sole impressions and 851 were retained; remaining 15 duplicates were found in other classes.
   - **5,741** unique images retained across all 8 classes.
2. **Subject Independence:** Filenames follow `cluster_<class_id>_<image_num>.BMP`. No donor IDs, patient identifiers, or fingers-per-subject metadata exist. True subject-independent evaluation is **UNVERIFIABLE** on this dataset; image-level deduplication was enforced to eliminate exact duplicate leakage.
3. **Class Distribution:** The deduplicated dataset contains **5,741 unique images** across all 8 ABO/Rh blood groups (A+: 565, A-: 764, AB+: 705, AB-: 760, B+: 648, B-: 741, O+: 851, O-: 707).

### Stratified Splits (70% / 15% / 15%)
Splits are locked via fixed random seed (`seed=42`) in [data_splitter.py](file:///c:/Users/Kartik/Downloads/fingerprint-based-blood-group-detection/data_splitter.py):
- **Training Set (N = 4,018):** [dataset_train_split.csv](file:///c:/Users/Kartik/Downloads/fingerprint-based-blood-group-detection/dataset/processed/dataset_train_split.csv) — *Feature scalers and PCA are fitted strictly here.*
- **Validation Set (N = 861):** [dataset_val_split.csv](file:///c:/Users/Kartik/Downloads/fingerprint-based-blood-group-detection/dataset/processed/dataset_val_split.csv) — *Ensemble weights ($w_1, w_2, w_3, w_4$) and calibration are optimized here.*
- **Held-Out Test Set (N = 862):** [dataset_test_split.csv](file:///c:/Users/Kartik/Downloads/fingerprint-based-blood-group-detection/dataset/processed/dataset_test_split.csv) — *Evaluated strictly once for final reporting.*
- **Sum Verification:** $4,018 + 861 + 862 = 5,741$ (Exact 100% split match).
- **Leakage Verification:** Zero MD5 hash overlap and zero base-stem overlap exist between Train, Val, and Test splits.

---

## Verified Benchmark Results (Held-Out Test Set)

Evaluated on the **entire 862 held-out test manifest** without sample exclusion ([model_comparison.csv](file:///c:/Users/Kartik/Downloads/fingerprint-based-blood-group-detection/models/model_comparison.csv), per-sample predictions saved in [test_predictions.csv](file:///c:/Users/Kartik/Downloads/fingerprint-based-blood-group-detection/models/test_predictions.csv)). Confidence intervals are computed via non-parametric empirical bootstrapping ($B = 1,000$, $\alpha = 0.05$). Paired CIs represent $\Delta = \text{Acc}_{\text{Ensemble}} - \text{Acc}_{\text{Model}}$.

| Model Architecture | Test Accuracy | 95% Bootstrap CI | Macro F1 | Weighted F1 | Correct / 862 | Paired Diff vs Ens (95% CI) | Beat Baseline? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Calibrated Weighted Ensemble (DL + ML)** | **91.88%** | **[89.9%, 93.6%]** | **91.80%** | **91.87%** | **792 / 862** | **Reference (0.00%)** | **Yes** |
| **Hard Majority Consensus Voting** | 89.79% | [87.6%, 91.8%] | 89.71% | 89.78% | 774 / 862 | +2.09% [+0.81%, +3.60%] | **Yes** |
| **Hybrid ResNet-SVM** | 88.17% | [85.8%, 90.1%] | 87.87% | 87.93% | 760 / 862 | +3.71% [+1.86%, +5.68%] | **Yes** |
| **Classical SVM (HOG Features)** | 87.12% | [84.6%, 89.3%] | 87.08% | 87.10% | 751 / 862 | +4.76% [+2.90%, +6.84%] | **Yes** |
| **Hybrid VGG16 + Priors + SVM** | 79.47% | [76.7%, 82.1%] | 79.45% | 79.38% | 685 / 862 | +12.41% [+9.98%, +15.08%] | **Yes** |
| **ResNet-50 Deep CNN (Pure Softmax)** | 62.53% | [59.3%, 65.7%] | 63.55% | 62.96% | 539 / 862 | +29.35% [+25.99%, +32.72%] | **Yes** |
| *Majority Class Baseline (O+)* | *14.85%* | — | — | — | *128 / 862* | — | — |
| *Random Guess Baseline* | *12.50%* | — | — | — | *108 / 862* | — | — |

### Primary Per-Class Performance ([classification_report.csv](file:///c:/Users/Kartik/Downloads/fingerprint-based-blood-group-detection/models/classification_report.csv))
*Evaluated on the Calibrated Weighted Ensemble over all 862 test samples:*

| Blood Group | Precision | Recall | F1-Score | Support (Actual Count) | Correct Predictions |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **A+** | 93.67% | 87.06% | 90.24% | 85 | 74 / 85 |
| **A-** | 94.50% | 89.57% | 91.96% | 115 | 103 / 115 |
| **AB+** | 91.92% | 85.85% | 88.78% | 106 | 91 / 106 |
| **AB-** | 91.23% | 91.23% | 91.23% | 114 | 104 / 114 |
| **B+** | 86.24% | **96.91%** | 91.26% | 97 | 94 / 97 |
| **B-** | **97.25%** | 95.50% | **96.36%** | 111 | 106 / 111 |
| **O+** | 90.84% | 92.97% | 91.89% | 128 | 119 / 128 |
| **O-** | 90.18% | 95.28% | 92.66% | 106 | 101 / 106 |
| **Overall / Macro** | **91.98%** | **91.79%** | **91.80%** | **862** | **792 / 862** |

---

## Supplementary Balanced Subset Analysis

To examine unweighted performance across classes independent of natural split frequencies, a balanced subset was drawn via fixed seed (`seed=42`) with **exactly 30 images per class ($N = 240$)** ([supplementary_balanced_comparison.csv](file:///c:/Users/Kartik/Downloads/fingerprint-based-blood-group-detection/models/supplementary_balanced_comparison.csv)):

| Model Architecture | Balanced Test Acc | 95% Bootstrap CI | Macro F1 | Weighted F1 | Correct / 240 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Calibrated Weighted Ensemble** | **92.50%** | **[89.2%, 95.8%]** | **92.47%** | **92.47%** | **222 / 240** |
| **Hard Majority Consensus Voting** | 90.00% | [86.2%, 93.8%] | 89.99% | 89.99% | 216 / 240 |
| **Hybrid ResNet-SVM** | 87.50% | [83.3%, 91.7%] | 87.34% | 87.34% | 210 / 240 |
| **Classical SVM (HOG Features)** | 87.50% | [83.7%, 91.7%] | 87.48% | 87.48% | 210 / 240 |
| **Hybrid VGG16 + Priors + SVM** | 78.33% | [73.3%, 83.8%] | 78.57% | 78.57% | 188 / 240 |
| **ResNet-50 Deep CNN** | 60.83% | [55.0%, 66.7%] | 61.17 | 61.17% | 146 / 240 |

---

## Scientific & Clinical Limitations

> [!WARNING]
> **Essential Clinical Reality:**
> Machine learning models trained on public fingerprint datasets may identify subtle image acquisition signatures, scanner artifacts, or synthetic distribution features present in the dataset.
> **There is no established, peer-reviewed biological mechanism connecting dermatoglyphic ridge formation (fetal development during weeks 10–24 of gestation) to ABO carbohydrate antigens or Rh(D) protein expression on erythrocyte membranes.**
> Outputs from this system are statistical machine learning estimates and **must never be used for blood transfusions, emergency typing, or clinical diagnosis**. Standard serological agglutination testing is mandatory.

---

## Cryptographic Security vs. Medical Reality

The repository includes an authenticated **AES-256-GCM Secure Vault** accessible via `/api/secure/*`:
- **What it verifies:** Cryptographic authentication guarantees **data confidentiality and tamper detection**. It ensures that biometric records and predicted logs stored on disk or transmitted over the network have not been altered or intercepted.
- **What it does NOT verify:** A mathematically valid cryptographic signature **does not prove the biological accuracy** of a prediction.

---

## USB Fingerprint Scanner Integration & Hardware Guide

This project features a modular multi-scanner biometric architecture supporting two physical USB fingerprint scanners alongside a high-fidelity virtual sensor mode.

### Architectural Overview

```
                      ┌──────────────────────────────────────┐
                      │    Flask Web Application (app.py)    │
                      └──────────────────┬───────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         │   ScannerManager (Singleton)   │
                         └───────┬───────────────┬───────┘
                                 │               │
        ┌────────────────────────┼───────────────┴────────────────────────┐
        ▼                        ▼                                        ▼
┌─────────────────┐    ┌─────────────────┐                      ┌─────────────────┐
│ Scanner1Adapter │    │ Scanner2Adapter │                      │SimulatedAdapter │
│ (Slot 1 USB)    │    │ (Slot 2 USB)    │                      │(Virtual Sensor) │
└────────┬────────┘    └────────┬────────┘                      └────────┬────────┘
         │                      │                                        │
         │ (Vendor C DLL/HTTP)  │ (Vendor C DLL/HTTP)                    │ (Sinusoidal/
         ▼                      ▼                                        ▼  Sample Pool)
   Physical USB 1         Physical USB 2                           Virtual Platen
         │                      │                                        │
         └──────────────────────┼────────────────────────────────────────┘
                                ▼
               ┌─────────────────────────────────┐
               │  normalizer.py (Canonical 512)  │
               └────────────────┬────────────────┘
                                ▼
         ┌─────────────────────────────────────────────┐
         │ Existing Preprocessing -> VGG16 -> SVM ->   │
         │ Bayesian Population Prevalence Calibration  │
         └─────────────────────────────────────────────┘
```

1. **Hot Device Switching:** Users can switch between **Scanner 1**, **Scanner 2**, and **Simulated Scanner** instantly from the web interface dropdown without restarting Flask.
2. **Strict Hardware Verification:** Non-biometric USB peripherals (e.g. webcams, Bluetooth adapters, keyboards, mice) are explicitly detected and flagged as unsupported. Non-biometric devices cannot be used as fingerprint scanners.
3. **Canonical Normalization:** Regardless of physical sensor platen size or DPI, captures are normalized to 512×512 8-bit grayscale with contrast balancing and Laplacian edge sharpness assessment.
4. **Direct Model Compatibility:** Normalized captures are fed directly into the existing `extract_ridge_domain_feature_vector()` and `preprocess_image_for_vgg16()` pipelines. **No retraining is required.**
5. **Driver Safety Guarantee:** The code does not simulate fake hardware or claim unverified SDKs function. If a scanner is plugged in but its vendor SDK is missing, the system displays the detected hardware VID/PID along with precise download links and instructions.

---

### How to Identify Your USB Scanners (Brand, Model, VID, PID)

If you own two USB fingerprint scanners and do not yet know their brand or model, follow these steps to identify them:

#### Method 1: Web UI Hardware Inspector (Easiest)
1. Start the Flask application: `python app.py`
2. Navigate to [http://localhost:5000](http://localhost:5000)
3. Under the **Fingerprint Image Input** card, click the **"Live USB Scanner"** tab.
4. Expand **"Hardware Inspector: Detected USB Peripherals"**.
5. Any recognized biometric scanner candidate will be marked with a green **Biometric Candidate** badge displaying its detected Name, Vendor ID, and Product ID.

#### Method 2: Windows PowerShell
Plug in your scanners and run in PowerShell:
```powershell
Get-CimInstance Win32_PnPEntity -Filter "DeviceID LIKE 'USB%VID%'" | Select-Object Name, DeviceID, Manufacturer, Status | Format-Table -AutoSize
```
Look for lines containing `USB\VID_xxxx&PID_yyyy`. The 4 characters after `VID_` represent the Vendor ID, and the 4 characters after `PID_` represent the Product ID.

#### Method 3: Python Discovery Command
```bash
python -c "from scanners import USBDetector; [print(f'Device: {d.name} | VID: 0x{d.vid} | PID: 0x{d.pid} | Biometric: {d.is_biometric_candidate} | Vendor: {d.vendor_name}') for d in USBDetector.detect_devices()]"
```

---

### Hardware Reference Lookup Table

Match your detected Vendor ID (VID) and Product ID (PID) against the known biometric vendor database:

| Manufacturer | Vendor ID (VID) | Common Product IDs (PID) | Typical Models | Official SDK / Driver Name |
| :--- | :---: | :---: | :--- | :--- |
| **Mantra Softech** | `0x27D0` | `0x0001`, `0x0002`, `0x0003` | MFS100, MIS100V2 | Mantra MFS100 Driver & Client Service / RD Service |
| **SecuGen Corp.** | `0x1162` | `0x1000`, `0x0330`, `0x0320` | Hamster Pro 20, Hamster Plus, Hamster IV | SecuGen WBF Driver & FDx SDK Pro for Windows (`sgfplib.dll`) |
| **DigitalPersona / HID** | `0x05BA`, `0x08FF` | `0x000A`, `0x000B`, `0x000C` | U.are.U 4500, 5160, 5300 | DigitalPersona U.are.U Windows SDK (`dpfpdd.dll`) |
| **IDEMIA / Morpho** | `0x079B` | `0x0024`, `0x0045`, `0x0046` | MorphoSmart MSO 1300 E2 / E3 | MorphoSmart Host SDK / Morpho RD Service (`MSO100.dll`) |
| **Futronic Technology** | `0x1491` | `0x0088`, `0x0090`, `0x0025` | FS80, FS88, FS90 | Futronic Windows API SDK (`ftrScanAPI.dll`) |
| **ZKTeco Inc.** | `0x1B55` | `0x0120`, `0x0130`, `0x0140` | SLK20R, ZK9500, Live20R | ZKTeco Live Fingerprint SDK (`zkfp.dll`) |
| **Startek** | `0x0A5C` | `0xE000`, `0xE001` | FM220, FM220U | Startek FM220 SDK (`libfpapi.dll`) |
| **Suprema Inc.** | `0x16D1` | `0x0400`, `0x0401` | BioMini, BioMini Plus 2 | Suprema BioMini SDK (`UFScanner.dll`) |

---

### Step-by-Step Vendor Driver & SDK Installation

Once your scanner's VID/PID matches one of the vendors above:

#### 1. Mantra MFS100 (`VID_27D0`)
1. Download official driver and client service from [Mantra Download Portal](https://download.mantratecapp.com/).
2. Run `MFS100Driver.exe` and `MFS100ClientService.exe` as Administrator.
3. Plug in the scanner. The Mantra service runs a local HTTP RD server on `http://127.0.0.1:11100/rd/capture`.
4. In `scanners/scanner1_adapter.py`, configure the `_capture_from_hardware_sdk` hook to query the local RD Service or load `MFS100.dll`.

#### 2. SecuGen Hamster (`VID_1162`)
1. Download SecuGen Windows drivers and **FDx SDK Pro for Windows** from [SecuGen Drivers](https://secugen.com/drivers/).
2. Install the WBF driver and copy `sgfplib.dll` to `C:\Windows\System32\` or the project root.
3. The adapter automatically detects `sgfplib.dll` presence and marks the device as **Ready**.
4. In `scanners/scanner1_adapter.py` (or `scanner2_adapter.py`), invoke `SGFPM_Create() -> SGFPM_Init() -> SGFPM_OpenDevice() -> SGFPM_GetImage()`.

#### 3. DigitalPersona U.are.U (`VID_05BA` / `VID_08FF`)
1. Download the DigitalPersona U.are.U Runtime Environment (RTE) and SDK from [HID Global Drivers](https://www.hidglobal.com/drivers).
2. Install the 64-bit runtime (`dpfpdd.dll`).
3. Ensure `dpfpdd.dll` is located in `C:\Program Files\DigitalPersona\U.are.U SDK\Windows\Lib\x64\` or system PATH.
4. Call `dpfpdd_init()` and `dpfpdd_capture()` in the adapter hook.

---

### Connecting Real Hardware to the Modular Adapters

Each scanner slot has its own dedicated adapter file:
- **Scanner 1:** [scanners/scanner1_adapter.py](file:///c:/Users/Kartik/Downloads/fingerprint-based-blood-group-detection/scanners/scanner1_adapter.py)
- **Scanner 2:** [scanners/scanner2_adapter.py](file:///c:/Users/Kartik/Downloads/fingerprint-based-blood-group-detection/scanners/scanner2_adapter.py)

To enable live hardware capture once the vendor SDK is installed, edit the `_capture_from_hardware_sdk` method:

```python
# Example for Mantra MFS100 via Local RD Service
import requests
import xml.etree.ElementTree as ET
from PIL import Image
import io
import base64

def _capture_from_hardware_sdk(self, dev, timeout_seconds):
    # Query local RD service endpoint
    capture_xml = '<PidOptions ver="1.0"><Opts fCount="1" fType="2" iCount="0" pCount="0" format="0" pidVer="2.0" timeout="10000" /></PidOptions>'
    res = requests.post("http://127.0.0.1:11100/rd/capture", data=capture_xml, timeout=timeout_seconds)
    root = ET.fromstring(res.text)
    # Extract raw base64 biometric image and return as PIL Image
    b64_data = root.find(".//Data").text
    raw_bytes = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(raw_bytes))
```

The captured image is automatically passed to `normalizer.normalize_fingerprint_image()`, which scales, centers, and balances contrast to 512×512 before sending to the VGG16 + SVM pipeline.

---

### Simulated Testing Mode

You can thoroughly test the entire interface, device switching, capture triggers, and prediction pipeline **without physical scanners** using the **Simulated Scanner Mode**:
- Enabled by default in the scanner dropdown.
- Cycles through realistic 500 DPI fingerprint impressions from the dataset or synthesizes anatomical ridge whorls.
- Computes real-time Laplacian sharpness, Michelson contrast, and biometric quality index (0–100).
- Automatically triggers the full VGG16 deep feature extractor, dermatoglyphic ridge priors, and Bayesian calibrated SVM classifier.

---

## Reproducible Setup & Execution Commands

### 1. Environment Installation
```bash
python -m pip install -r requirements.txt
```

### 2. Dataset Audit & Stratified Manifest Generation
Generates deduplicated 70% / 15% / 15% train, validation, and test splits with zero overlap:
```bash
python data_splitter.py
```

### 3. Run Comprehensive Evaluation Benchmark
Evaluates the entire 862 held-out test manifest, fits ensemble weights on validation data, calculates 95% bootstrap CIs, and exports reports and per-sample predictions:
```bash
python run_comprehensive_audit_and_eval.py
```

### 4. Independent Dataset & Split Disjointness Audit
Verifies exact counts (6,321 -> 580 removed -> 5,741 unique), bracket files (1,417), split sums (4,018 + 861 + 862 = 5,741), hash disjointness, and stem independence:
```bash
python audit_dataset_and_splits.py
```

### 5. Independent Metrics & Confidence Interval Verification
Recomputes accuracy, macro/weighted F1, per-class metrics, confusion matrices, and paired bootstrap CIs directly from saved `models/test_predictions.csv`:
```bash
python audit_recompute_predictions.py
```

### 6. Single-Image CLI Prediction
```bash
python predict.py "sample dataset/sample_data.jpg"
```

### 7. Run USB Scanner Subsystem & API Test Suite
Executes the comprehensive automated unit and integration test suite verifying device discovery, simulated captures, hot device switching, driver safety verification, and end-to-end ML prediction pipeline integration:
```bash
python test_scanners.py
```

### 8. Launch the Web Application
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser.

