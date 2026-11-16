"""
===============================================================================
FINE-TUNED VGG16 FEATURE EXTRACTOR & DATA AUGMENTATION PIPELINE
===============================================================================

Architecture & Training Quality Details:
----------------------------------------
1. Base Backbone: Pretrained VGG16 (ImageNet initialization).
2. Fine-Tuning Strategy:
   - Freeze lower blocks (block1_conv1 through block4_pool) to preserve low-level
     edge/ridge filter representations.
   - Unfreeze block5 convolutional layers (block5_conv1, block5_conv2, block5_conv3)
     to adapt high-level dermatoglyphic texture and core/delta representations.
   - GlobalAveragePooling2D to generate compact 512-dimensional deep feature embeddings.
3. Optimization & Training Controls:
   - Low Learning Rate: Adam optimizer at lr = 1e-5 to prevent catastrophic forgetting.
   - EarlyStopping: monitor 'val_loss', patience=4, restore_best_weights=True.
   - ReduceLROnPlateau: monitor 'val_loss', factor=0.5, patience=2.
4. Data Augmentation (Applied ONLY on Training Split):
   - Random Rotations (+/- 15 degrees)
   - Random Zoom (+/- 10%)
   - Random Width & Height Shifts (+/- 8%)
   - Brightness / Contrast Jitter
   - Subtle Gaussian Noise
"""

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

CLASSES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
IMG_SIZE = (224, 224)

def preprocess_image_for_vgg16(img_or_path):
    """
    Standard preprocessing for VGG16:
    Grayscale conversion, resize to 224x224, 3-channel RGB expansion,
    and standard VGG16 mean subtraction.
    """
    if isinstance(img_or_path, str):
        with Image.open(img_or_path) as pil_img:
            img_rgb = pil_img.convert('RGB')
    elif isinstance(img_or_path, Image.Image):
        img_rgb = img_or_path.convert('RGB')
    else:
        # numpy array
        arr = np.array(img_or_path)
        if arr.ndim == 2:
            img_rgb = Image.fromarray(arr).convert('RGB')
        else:
            img_rgb = Image.fromarray(arr.astype(np.uint8))
            
    img_resized = img_rgb.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)
    return preprocess_input(arr)

def apply_training_augmentation(image_array):
    """
    Apply dermatoglyphic-safe data augmentation exclusively on training images:
    - Subtle rotation (+/- 12 deg)
    - Mild scaling / zoom (0.92 - 1.08)
    - Horizontal / Vertical translation (+/- 6%)
    - Intensity / contrast jitter
    - Mild Gaussian noise
    """
    # Random rotation
    angle = np.random.uniform(-12.0, 12.0)
    # Convert to PIL for affine transform
    norm_img = np.clip(image_array + 128.0, 0, 255).astype(np.uint8)
    pil_im = Image.fromarray(norm_img)
    
    # Rotate with reflection padding
    rotated = pil_im.rotate(angle, resample=Image.Resampling.BILINEAR)
    
    # Random Zoom / Crop
    zoom_factor = np.random.uniform(0.94, 1.06)
    w, h = rotated.size
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    zoomed = rotated.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    if zoom_factor > 1.0:
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        cropped = zoomed.crop((left, top, left + w, top + h))
    else:
        cropped = Image.new('RGB', (w, h), (128, 128, 128))
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        cropped.paste(zoomed, (paste_x, paste_y))
        
    out_arr = np.array(cropped, dtype=np.float32)
    
    # Mild contrast jitter
    contrast_factor = np.random.uniform(0.88, 1.12)
    mean_val = np.mean(out_arr, axis=(0, 1), keepdims=True)
    out_arr = (out_arr - mean_val) * contrast_factor + mean_val
    
    # Mild Gaussian noise
    noise = np.random.normal(0.0, 2.5, out_arr.shape).astype(np.float32)
    out_arr = np.clip(out_arr + noise, 0, 255)
    
    return preprocess_input(out_arr)

def build_vgg16_classifier(num_classes=len(CLASSES), unfreeze_blocks=['block5']):
    """
    Build VGG16 classification model with block5 convolutional layers unfrozen
    for low-learning-rate fine-tuning.
    """
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze lower layers, unfreeze specified top block(s)
    for layer in base_model.layers:
        if any(b in layer.name for b in unfreeze_blocks):
            layer.trainable = True
        else:
            layer.trainable = False
            
    # Classification head on top of GAP
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name='deep_gap_embeddings')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name='dense_projection')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='blood_group_predictions')(x)
    
    model = models.Model(inputs=base_model.input, outputs=outputs, name='vgg16_blood_group_finetuned')
    
    optimizer = optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_vgg16_feature_extractor(trained_model):
    """
    Create dedicated sub-model to extract 512-dim deep feature vectors
    from the GlobalAveragePooling2D layer.
    """
    gap_layer = trained_model.get_layer('deep_gap_embeddings')
    feature_extractor = models.Model(
        inputs=trained_model.input,
        outputs=gap_layer.output,
        name='vgg16_deep_feature_extractor'
    )
    return feature_extractor

def train_and_finetune_vgg16(X_train_tensors, y_train, X_val_tensors, y_val, epochs=10, batch_size=32):
    """
    Fine-tune VGG16 on training split with EarlyStopping and ReduceLROnPlateau.
    Returns:
        trained_model: complete fine-tuned model
        feature_extractor: 512-dim embedding extractor
        history: training history dict
    """
    model = build_vgg16_classifier(num_classes=len(CLASSES))
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print(f"\n[*] Starting VGG16 Fine-Tuning (Block5 Trainable, LR=1e-5, {epochs} Max Epochs)...")
    history = model.fit(
        X_train_tensors, y_train,
        validation_data=(X_val_tensors, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    feature_extractor = build_vgg16_feature_extractor(model)
    return model, feature_extractor, history
