import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np
import os

# --- CONFIGURATION ---
# CRITICAL FIX: Point directly to the folder containing the breed subfolders
# Try 'dataset/test' first. If you don't have a test folder, use 'dataset/train'.
DATASET_PATH = os.path.join('dataset', 'train') 
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def generate_report():
    # 1. Load Model
    if not os.path.exists('cow_model.h5'):
        print("❌ Error: 'cow_model.h5' not found.")
        return
    
    print(f"⏳ Loading images from: {os.path.abspath(DATASET_PATH)}")
    
    # Check if the path exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: The folder '{DATASET_PATH}' does not exist.")
        print("   Please check if your folder is named 'dataset' or 'Dataset'.")
        return

    # 2. Setup Generator (No split needed now)
    datagen = ImageDataGenerator()

    # 3. Create Generator
    try:
        test_generator = datagen.flow_from_directory(
            DATASET_PATH,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False  # CRITICAL: Keeps predictions aligned with labels
        )
    except Exception as e:
        print(f"❌ Error initializing generator: {e}")
        return

    # Check if classes were found
    if test_generator.num_classes < 2:
        print("\n⚠️ ERROR: Found fewer than 2 classes.")
        print("   Make sure DATASET_PATH points to the folder holding the breed names.")
        print("   Correct structure: dataset/train/Gir, dataset/train/Sahiwal, etc.")
        return

    print("🧠 Model Loaded. Running predictions...")
    model = tf.keras.models.load_model('cow_model.h5')

    # 4. Predict
    Y_pred = model.predict(test_generator, verbose=1)
    y_pred_classes = np.argmax(Y_pred, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # 5. Print Report
    print("\n" + "="*60)
    print("📊 CLASSIFICATION REPORT")
    print("="*60)
    
    # Generate the table
    report = classification_report(y_true, y_pred_classes, target_names=class_labels)
    print(report)
    print("="*60)

if __name__ == "__main__":
    generate_report()