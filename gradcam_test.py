import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = "dr_model.keras"
IMAGE_SIZE = (224, 224)
LAST_CONV_LAYER = "conv2d_2"

model = tf.keras.models.load_model(MODEL_PATH)


def make_gradcam(image_path):
    # Load and prepare image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    resized = image.resize(IMAGE_SIZE)

    array = np.asarray(resized, dtype=np.float32)
    input_image = np.expand_dims(array, axis=0)

    # Build an intermediate model that returns every layer output
    layer_outputs = []
    x = tf.keras.Input(shape=(224, 224, 3))
    current = x

    for layer in model.layers:
        current = layer(current)
        layer_outputs.append(current)

    grad_model = tf.keras.Model(
        inputs=x,
        outputs=[
            layer_outputs[
                [layer.name for layer in model.layers].index(LAST_CONV_LAYER)
            ],
            current
        ]
    )

    # Calculate gradients
    input_tensor = tf.convert_to_tensor(input_image)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            input_tensor, training=False
        )
        class_index = tf.argmax(predictions[0])
        score = predictions[:, class_index]

    gradients = tape.gradient(score, conv_outputs)

    if gradients is None:
        raise ValueError(
            "Gradients unavailable. Please share this error."
        )

    # Create Grad-CAM heatmap
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    feature_maps = conv_outputs[0]

    heatmap = tf.reduce_sum(
        feature_maps * weights[0][tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)

    if float(maximum) > 0:
        heatmap = heatmap / maximum

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis], IMAGE_SIZE
    ).numpy().squeeze()

    # Overlay heatmap on the retinal image
    colour_map = plt.get_cmap("jet")
    coloured = (
        colour_map(heatmap)[..., :3] * 255
    ).astype(np.uint8)

    base = np.asarray(resized, dtype=np.float32)
    overlay = np.clip(
        0.55 * base + 0.45 * coloured, 0, 255
    ).astype(np.uint8)

    output_path = Path("static") / "gradcam_test.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(overlay).resize(original_size).save(output_path)

    print("Grad-CAM created:", output_path.resolve())
    print("Predicted class index:", int(class_index))
    print("Prediction scores:", predictions.numpy()[0])


if __name__ == "__main__":
    test_folder = Path("static/uploads")

    images = [
        p for p in test_folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if not images:
        print("First upload a retinal image using your website.")
    else:
        make_gradcam(images[-1])