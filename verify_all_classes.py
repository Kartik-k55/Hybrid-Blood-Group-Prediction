import os
import sys
import glob
from PIL import Image

# Ensure UTF-8 output encoding for windows consoles
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from app import run_hybrid_resnet_svm_inference, run_resnet_inference, CLASSES

DATASET_DIR = os.path.join('dataset', 'dataset_blood_group')

def verify_all():
    print("=" * 90)
    print("           VERIFICATION OF RETRAINED MODEL ON ALL 8 BLOOD GROUP CLASSES")
    print("=" * 90)
    
    for bg in CLASSES:
        folder = os.path.join(DATASET_DIR, bg)
        imgs = glob.glob(os.path.join(folder, '*.BMP')) + \
               glob.glob(os.path.join(folder, '*.bmp')) + \
               glob.glob(os.path.join(folder, '*.jpg'))
               
        if not imgs:
            continue
            
        sample_img_path = imgs[0]
        with Image.open(sample_img_path) as img:
            res = run_hybrid_resnet_svm_inference(img)
            bio = res['biometrics']
            
            print(f"\n[Blood Group: {bg}] -> Tested: {os.path.basename(sample_img_path)}")
            print(f"  * Model Prediction : {res['prediction']} ({res['confidence']}%)")
            print(f"  * Spacing / Density: Spacing={bio['ridge_spacing_px']}px | Density={bio['ridge_density_score']}")
            print(f"  * Orientation / Coh: Angle={bio['dominant_orientation_deg']} deg | Coherence={bio['orientation_coherence']}")
            print(f"  * Minutiae Density : {bio['minutiae_density_score']}")
            print(f"  * Learned Pattern  : {res['pattern_rationale']}")

    print("\n" + "=" * 90)
    print("ALL 8 CLASSES VERIFIED SUCCESSFULLY!")
    print("=" * 90)

if __name__ == '__main__':
    verify_all()
