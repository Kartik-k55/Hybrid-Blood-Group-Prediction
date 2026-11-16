import os
import glob
from PIL import Image
from app import run_hybrid_vgg16_svm_inference, run_hybrid_resnet_svm_inference, run_resnet_inference, run_svm_inference, CLASSES

print("Testing all 4 model pipelines across all 8 Blood Group Classes:\n")
for bg in CLASSES:
    folder = os.path.join('dataset', 'dataset_blood_group', bg)
    imgs = glob.glob(os.path.join(folder, '*.BMP')) + glob.glob(os.path.join(folder, '*.bmp'))
    if not imgs:
        continue
    img = Image.open(imgs[0])
    r_vgg = run_hybrid_vgg16_svm_inference(img)
    r_res_svm = run_hybrid_resnet_svm_inference(img)
    r_res = run_resnet_inference(img)
    r_svm = run_svm_inference(img)
    print(f"True: {bg:4s} | VGG16_Hybrid: {r_vgg['prediction']:4s} ({r_vgg['confidence']:.1f}%) | ResNet_SVM: {r_res_svm['prediction']:4s} ({r_res_svm['confidence']:.1f}%) | ResNet: {r_res['prediction']:4s} ({r_res['confidence']:.1f}%) | SVM: {r_svm['prediction']:4s} ({r_svm['confidence']:.1f}%)")
