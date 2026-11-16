import os
import sys
import glob
from PIL import Image

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from app import run_all_models_comparison, CLASSES

print("=" * 110)
print("       CROSS-MODEL COMPARISON BENCHMARK ACROSS ALL 8 BLOOD GROUP SAMPLES")
print("=" * 110)

for bg in CLASSES:
    folder = os.path.join('dataset', 'dataset_blood_group', bg)
    imgs = glob.glob(os.path.join(folder, '*.BMP')) + glob.glob(os.path.join(folder, '*.bmp'))
    if not imgs:
        continue
    img = Image.open(imgs[0])
    res = run_all_models_comparison(img)
    models = res['models_comparison']
    
    vgg_p = next(m for m in models if m['id'] == 'vgg16_hybrid')
    res_svm_p = next(m for m in models if m['id'] == 'hybrid_svm')
    resnet_p = next(m for m in models if m['id'] == 'resnet')
    svm_p = next(m for m in models if m['id'] == 'svm')
    
    print(f"\n[TRUE: {bg:4s}] Sample: {os.path.basename(imgs[0])}")
    print(f"  * [Rank 1] Hybrid VGG16 (Recommended): {vgg_p['prediction']:4s} ({vgg_p['confidence']:.1f}%)")
    print(f"  * [Rank 2] Hybrid ResNet-SVM          : {res_svm_p['prediction']:4s} ({res_svm_p['confidence']:.1f}%)")
    print(f"  * [Rank 3] ResNet-34 CNN               : {resnet_p['prediction']:4s} ({resnet_p['confidence']:.1f}%)")
    print(f"  * [Rank 4] Classical SVM (HOG)         : {svm_p['prediction']:4s} ({svm_p['confidence']:.1f}%)")
    print(f"  * Consensus Result                   : {res['consensus_group']} ({res['agreement_count']}, {res['agreement_percentage']}%)")

print("\n" + "=" * 110)
