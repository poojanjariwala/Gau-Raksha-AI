import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import json
import os
import sys

# --- CONFIGURATION ---
DATA_DIR = 'dataset/train'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Smaller batch size for CPU stability
EPOCHS = 30      # We don't need 50 if we fine-tune correctly

def train_pro_model():
    print("\n--- 🚀 STARTING PRO TRAINING PROTOCOL (EfficientNetB0) ---\n")

    # 1. INTELLIGENT DATA AUGMENTATION
    # EfficientNet expects 0-255 values, so we DO NOT use rescale=1./255
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    print("⏳ Loading images...")
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation'
    )

    # Save Class Names
    class_names = {v: k for k, v in train_generator.class_indices.items()}
    with open('class_indices.json', 'w') as f:
        json.dump(class_names, f)

    # 2. BUILD PRO MODEL (EfficientNetB0)
    # This model is much smarter at fine-grained details (like hump shape)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # --- CRITICAL STEP: UNFREEZE TOP LAYERS ---
    # We unfreeze the last 20 layers so the model adapts to "Cows" specifically
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)  # Higher dropout to prevent memorizing
    predictions = Dense(len(class_names), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 3. COMPILE WITH LOW LEARNING RATE
    # Low rate is crucial when fine-tuning to not "break" the pre-trained weights
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. TRAIN
    print("\n⚡ Training High-Accuracy Model... (This will take time)")
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator, callbacks=callbacks)
    
    model.save('cow_model.h5')
    print("\n🏆 DONE: High-Accuracy Model Saved!")

if __name__ == "__main__":
    train_pro_model()