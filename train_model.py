import os
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Replace this with your actual dataset folder path
DATASET_PATH = r"C:\Users\acer\Downloads\Dataset from fundus images for the study of diabetic retinopathy_V03"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_PATH}"
    )

# Load training images
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

# Load validation images
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)
print("Training class counts:")
import numpy as np

for i, name in enumerate(train_ds.class_names):
    count = sum(
        int(np.sum(labels.numpy() == i))
        for _, labels in train_ds
    )
    print(name, ":", count)

print("Validation class counts:")
for i, name in enumerate(val_ds.class_names):
    count = sum(
        int(np.sum(labels.numpy() == i))
        for _, labels in val_ds
    )
    print(name, ":", count)

class_names = train_ds.class_names
print("Training batches:", len(train_ds))
print("Validation batches:", len(val_ds))
print("Class names:", class_names)
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Build a simple CNN
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

# EfficientNetB0 includes its own input rescaling
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)




# Get labels from the training dataset
train_labels = np.concatenate([
    labels.numpy() for _, labels in train_ds
])

# Calculate class weights
classes = np.unique(train_labels)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_labels
)
class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("Class weights:", class_weights)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights
)

model.save("dr_model.keras")
print("Model saved as dr_model.keras")

