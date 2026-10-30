import tensorflow as tf
from pathlib import Path
import numpy as np

DATASET_PATH = Path(
    r"C:\Users\acer\Downloads\Dataset from fundus images for the study of diabetic retinopathy_V03"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    str(DATASET_PATH),
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(DATASET_PATH),
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

print("\nTraining class counts:")
for i, name in enumerate(train_ds.class_names):
    count = sum(
        int(np.sum(labels.numpy() == i))
        for _, labels in train_ds
    )
    print(name, ":", count)

print("\nValidation class counts:")
for i, name in enumerate(val_ds.class_names):
    count = sum(
        int(np.sum(labels.numpy() == i))
        for _, labels in val_ds
    )
    print(name, ":", count)