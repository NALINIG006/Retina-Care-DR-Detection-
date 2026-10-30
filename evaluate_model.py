import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = Path("dr_model.keras")
DATASET_PATH = Path(
    r"C:\Users\acer\Downloads\Dataset from fundus images for the study of diabetic retinopathy_V03"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_PATH}")

print("Loading model...")
model = tf.keras.models.load_model(str(MODEL_PATH))
print("Model output shape:", model.output_shape)

print("Loading validation images...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(DATASET_PATH),
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

class_names = val_ds.class_names

all_labels = np.concatenate([
    labels.numpy() for _, labels in val_ds
])

print("\nValidation class counts:")
print(np.bincount(all_labels, minlength=7))

print("\nPredicting...")
predictions = model.predict(val_ds)
y_pred = np.argmax(predictions, axis=1)
y_true = all_labels

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
    digits=3
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
))

print("\nEvaluation completed!")