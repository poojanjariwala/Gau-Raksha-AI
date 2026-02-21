import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import json
import os
import sys

# --- CONFIGURATION ---
DATA_DIR = 'dataset/train'  # <--- This must match your folder structure
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

def train_winning_model():
    print("\n--- 🚀 STARTING TRAINING PROTOCOL ---\n")

    # 1. VERIFY FOLDERS
    if not os.path.exists(DATA_DIR):
        print(f"❌ CRITICAL ERROR: I cannot find the folder '{DATA_DIR}'")
        print("   -> Make sure your folder structure is: Final_Cow_Project -> dataset -> train")
        sys.exit(1)

    # 2. IMAGE GENERATOR (Augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    print("⏳ Scanning for images...")
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation'
    )

    if train_generator.samples == 0:
        print("❌ ERROR: No images found! Check if your breed folders are empty.")
        sys.exit(1)

    # 3. SAVE CLASS NAMES
    class_names = {v: k for k, v in train_generator.class_indices.items()}
    with open('class_indices.json', 'w') as f:
        json.dump(class_names, f)
    print(f"✅ Found {len(class_names)} breeds. Saved names to 'class_indices.json'.")

    # 4. BUILD MODEL (Transfer Learning)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(len(class_names), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 5. TRAIN
    print("\n⚡ Training AI Brain... (This will take time)")
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    ]
    
    model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator, callbacks=callbacks)
    
    # 6. SAVE
    model.save('cow_model.h5')
    print("\n🏆 SUCCESS: Model saved as 'cow_model.h5'. You are ready to run the app!")

if __name__ == "__main__":
    train_winning_model()