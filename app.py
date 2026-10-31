from flask import Flask, render_template, request
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
TEMPLATE_FOLDER = BASE_DIR / "templates"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

MODEL_PATH = BASE_DIR / "dr_model.keras"
IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

DATASET_PATH = Path(
    r"C:\Users\acer\Downloads\Dataset from fundus images for the study of diabetic retinopathy_V03"
)

# Load the trained model
model = tf.keras.models.load_model(str(MODEL_PATH))

# IMPORTANT:
# These names MUST be in the same order used during model training.
# Confirm the original training class order before using this list.
class_names = [
    "1. No DR signs",
    "2. Mild (or early) NPDR",
    "3. Moderate NPDR",
    "4. Severe NPDR",
    "5. Very Severe NPDR",
    "6. PDR",
    "7. Advanced PDR",
]

if len(class_names) != model.output_shape[-1]:
    raise ValueError(
        f"Model has {model.output_shape[-1]} outputs, "
        f"but {len(class_names)} class names were provided."
    )
# Class-specific explanations
explanations = {
    0: {
        "severity": "No apparent DR signs",
        "description": (
            "The model did not identify signs of diabetic "
            "retinopathy in this image. This does not rule out "
            "eye disease. Clinical examination is recommended."
        ),
        "recommendation": (
            "Continue regular eye screening as advised by "
            "your eye-care professional."
        )
    },
    1: {
        "severity": "Mild NPDR predicted",
        "description": (
            "The model predicts the mild NPDR category. "
            "This is an AI prediction, not a confirmed diagnosis."
        ),
        "recommendation": (
            "Arrange an eye examination with a qualified "
            "ophthalmologist to confirm the findings."
        )
    },
    2: {
        "severity": "Moderate NPDR predicted",
        "description": (
            "The model predicts the moderate NPDR category. "
            "A clinical examination is needed to assess the retina."
        ),
        "recommendation": (
            "Consult an ophthalmologist for assessment and "
            "follow the recommended screening schedule."
        )
    },
    3: {
        "severity": "Severe NPDR predicted",
        "description": (
            "The model predicts the severe NPDR category. "
            "The result requires prompt clinical assessment."
        ),
        "recommendation": (
            "Arrange a prompt ophthalmology appointment to "
            "confirm the result and discuss appropriate care."
        )
    },
    4: {
        "severity": "Very severe NPDR predicted",
        "description": (
            "The model predicts the very severe NPDR category. "
            "This classification must be confirmed by an eye specialist."
        ),
        "recommendation": (
            "Seek prompt assessment from a qualified ophthalmologist."
        )
    },
    5: {
        "severity": "PDR predicted",
        "description": (
            "The model predicts the proliferative diabetic "
            "retinopathy category. This is not a confirmed diagnosis."
        ),
        "recommendation": (
            "Seek prompt ophthalmology assessment to confirm "
            "the findings and determine the next steps."
        )
    },
    6: {
        "severity": "Advanced PDR predicted",
        "description": (
            "The model predicts the advanced PDR category. "
            "An eye specialist must assess the image and clinical history."
        ),
        "recommendation": (
            "Seek prompt specialist eye assessment. Sudden vision "
            "loss requires urgent medical attention."
        )
    }
}

@app.route("/")

def home():
    return render_template("index.html")
def generate_gradcam(image_array, predicted_index):
    base_model = model.get_layer("efficientnetb0")
    last_conv_layer = base_model.get_layer("top_conv")

    # Create Grad-CAM feature model
    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    image_tensor = tf.convert_to_tensor(
        image_array, dtype=tf.float32
    )

    with tf.GradientTape() as tape:
        conv_features, base_output = feature_model(
            image_tensor, training=False
        )

        x = base_output

        # Pass through classification layers
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            if layer.name == "efficientnetb0":
                continue

            x = layer(x, training=False)

        class_score = x[:, predicted_index]

    # Calculate gradients
    gradients = tape.gradient(class_score, conv_features)

    if gradients is None:
        raise ValueError(
            "Grad-CAM gradients are None. Check model architecture."
        )

    weights = tf.reduce_mean(
        gradients, axis=(0, 1, 2)
    )

    features = conv_features[0]

    heatmap = tf.reduce_sum(
        features * weights, axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) +
        tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = np.power(heatmap, 2.0)

    # Convert heatmap to RGB
    heatmap_rgb = plt.get_cmap("jet")(
        heatmap
    )[:, :, :3]

    heatmap_rgb = np.uint8(heatmap_rgb * 255)

    # Original image
    original_image = Image.fromarray(
        np.uint8(image_array[0])
    ).convert("RGB").resize((224, 224))

    # Resize heatmap
    heatmap_coloured = Image.fromarray(
        heatmap_rgb
    ).resize(original_image.size)

    # Blend original image and heatmap
    overlay = Image.blend(
        original_image,
        heatmap_coloured,
        alpha=0.30
    )

    return overlay
    # Apply a colour map
    heatmap_rgb = plt.get_cmap("jet")(heatmap)[:, :, :3]
    heatmap_rgb = np.uint8(heatmap_rgb * 255)

    return Image.fromarray(heatmap_rgb)
@app.route("/predict", methods=["POST"])
def predict():
    print("PREDICT ROUTE STARTED", flush=True)
    if "image" not in request.files:
        return "Please select a retinal image.", 400

    file = request.files["image"]

    if not file.filename:
        return "No image selected.", 400

    filename = secure_filename(file.filename)
    extension = filename.rsplit(".", 1)[-1].lower()

    if "." not in filename or extension not in ALLOWED_EXTENSIONS:
        return "Upload a PNG, JPG, or JPEG image.", 400

    try:
        # Read and validate the uploaded image
        image = Image.open(file.stream).convert("RGB")
        image.load()

        # Prepare image for the model
        model_image = image.resize(IMAGE_SIZE)
        image_array = np.asarray(model_image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        # AI prediction
        predictions = model.predict(image_array, verbose=0)[0]
        print("PREDICTION COMPLETED")

        print("\n--- DEBUG: MODEL PREDICTIONS ---")
        for i, score in enumerate(predictions):
            print(f"{i}: {class_names[i]} = {score:.4f}")

        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        result_info = explanations[predicted_index]

        severity = result_info["severity"]
        description = result_info["description"]
        recommendation = result_info["recommendation"]

        print("Predicted index:", predicted_index)
        print("Predicted class:", predicted_class)
        # Severity and explanation for each retinal class
        class_details = {
            "1. No DR signs": {
                "severity": "No apparent DR signs",
                "description": (
                    "The model did not identify signs of diabetic "
                    "retinopathy in this image. This does not rule "
                    "out other eye conditions."
                ),
                "recommendation": (
                    "Continue regular diabetic eye screening as "
                    "recommended by your eye-care professional."
                )
            },
            "2. Mild (or early) NPDR": {
                "severity": "Mild",
                "description": (
                    "Early diabetic retinal changes may be present, "
                    "including small changes in retinal blood vessels."
                ),
                "recommendation": (
                    "Arrange an eye examination and follow the "
                    "screening schedule advised by your clinician."
                )
            },
            "3. Moderate NPDR": {
                "severity": "Moderate",
                "description": (
                    "More retinal blood-vessel abnormalities may "
                    "be present than in mild NPDR."
                ),
                "recommendation": (
                    "Consult an ophthalmologist to confirm the stage "
                    "and appropriate follow-up."
                )
            },
            "4. Severe NPDR": {
                "severity": "Severe",
                "description": (
                    "This category can indicate advanced retinal "
                    "blood-vessel changes and a higher risk of "
                    "progression."
                ),
                "recommendation": (
                    "Arrange prompt assessment by an ophthalmologist."
                )
            },
            "5. Very Severe NPDR": {
                "severity": "Very severe",
                "description": (
                    "This dataset category indicates a more advanced "
                    "non-proliferative stage, if the dataset label "
                    "definition confirms this meaning."
                ),
                "recommendation": (
                    "Seek prompt professional assessment to confirm "
                    "the findings and appropriate care."
                )
            },
            "6. PDR": {
                "severity": "Proliferative DR category",
                "description": (
                    "Proliferative diabetic retinopathy can involve "
                    "abnormal new blood-vessel growth and may threaten "
                    "vision."
                ),
                "recommendation": (
                    "Arrange prompt assessment by an ophthalmologist."
                )
            },
            "7. Advanced PDR": {
                "severity": "Advanced PDR category",
                "description": (
                    "This dataset category may represent advanced "
                    "proliferative disease. Its exact clinical meaning "
                    "must be confirmed against the dataset documentation."
                ),
                "recommendation": (
                    "Seek prompt ophthalmologist assessment. Do not "
                    "rely on this AI prediction alone."
                )
            }
        }

        # Look up details using the predicted class
        details = class_details.get(
            predicted_class,
            {
                "severity": "Unverified class",
                "description": (
                    "The predicted label does not exactly match "
                    "the configured class names. Verify the model's "
                    "training labels before interpreting this result."
                ),
                "recommendation": (
                    "Consult an ophthalmologist for professional "
                    "evaluation."
                )
            }
        )

        severity = details["severity"]
        description = details["description"]
        recommendation = details["recommendation"]
        confidence = float(predictions[predicted_index]) * 100

        # Save uploaded image with a unique filename
        import uuid
        saved_name = f"{uuid.uuid4().hex}.{extension}"
        save_path = UPLOAD_FOLDER / saved_name
        image.save(save_path)

        image_path = f"/static/uploads/{saved_name}"
        gradcam_image = generate_gradcam( image_array, predicted_index )
        print("GRADCAM COMPLETED")

        gradcam_name = f"gradcam_{saved_name.rsplit('.', 1)[0]}.jpg"
        gradcam_file = UPLOAD_FOLDER / gradcam_name

        gradcam_image.save(gradcam_file)

        gradcam_path = f"/static/uploads/{gradcam_name}"
        print("Prediction completed:", predicted_class)
        print("Confidence:", round(confidence, 2), "%")

        # Render the result page
        return render_template(
            "result.html",
            severity=severity,
            description=description,
            recommendation=recommendation,
            image_path=image_path,
            predicted_class=predicted_class,
            confidence=round(confidence, 2),
            gradcam_path=gradcam_path
        )

    except Exception as error:
        app.logger.exception("Image prediction failed")
        return (
            "Unable to analyse this image. Check the terminal for the error.",
            500
        )


@app.errorhandler(413)
def file_too_large(error):
    return "Image is too large. Maximum upload size is 10 MB.", 413


if __name__ == "__main__":
    app.run(debug=False)