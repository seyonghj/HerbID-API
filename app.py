"""
=============================================================
 HerbID Iloilo API
 Flask + TensorFlow + ResNet50V2
 Northern Iloilo State University - BSIT Capstone
=============================================================
"""

import os
import io
import json
import logging
from datetime import datetime

import numpy as np

from PIL import Image

from flask import (
    Flask,
    request,
    jsonify
)

from flask_cors import CORS

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input

import os

MODEL_PATH = "model/herb_resnet50v2.h5"

print("=" * 60)
print("Exists:", os.path.exists(MODEL_PATH))

if os.path.exists(MODEL_PATH):
    print("Size:", os.path.getsize(MODEL_PATH))

    with open(MODEL_PATH, "rb") as f:
        header = f.read(32)

    print("Header:", header)

print("=" * 60)

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "model/herb_resnet50v2.h5"
CLASS_FILE = "class_names.json"

IMG_WIDTH = 224
IMG_HEIGHT = 224

CONFIDENCE_THRESHOLD = 0.60

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}

MAX_IMAGE_SIZE = 10 * 1024 * 1024   # 10 MB

# ============================================================
# Flask
# ============================================================

app = Flask(__name__)

CORS(app)

app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE

# ============================================================
# Logging
# ============================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger("HerbID")

# ============================================================
# Load Classes
# ============================================================

def load_class_names():

    logger.info("Loading class labels...")

    if not os.path.exists(CLASS_FILE):
        raise FileNotFoundError(
            f"{CLASS_FILE} not found."
        )

    with open(CLASS_FILE, "r", encoding="utf-8") as f:

        data = json.load(f)

    if isinstance(data, dict):

        ordered = sorted(
            data.items(),
            key=lambda x: x[1]
        )

        classes = [c[0] for c in ordered]

    elif isinstance(data, list):

        classes = data

    else:

        raise ValueError(
            "Invalid class_names.json format."
        )

    logger.info(
        f"Loaded {len(classes)} herb classes."
    )

    return classes


CLASS_NAMES = load_class_names()

HERB_METADATA = {
    herb_name: {
        "scientificName": "",
        "family": "",
        "description": "",
    }
    for herb_name in CLASS_NAMES
}

# ============================================================
# Load TensorFlow Model
# ============================================================

logger.info("Loading TensorFlow model...")

MODEL = load_model(MODEL_PATH)


def warm_up_model():

    if MODEL is None:
        logger.warning("Model is not available for warm-up.")
        return

    try:
        dummy_input = np.zeros(
            (1, IMG_HEIGHT, IMG_WIDTH, 3),
            dtype=np.float32
        )
        dummy_input = preprocess_input(dummy_input)
        MODEL.predict(dummy_input, verbose=0)
        logger.info("Model warm-up completed.")
    except Exception as exc:
        logger.warning("Model warm-up failed: %s", exc)


warm_up_model()

logger.info("Model loaded successfully.")

# ============================================================
# Utilities
# ============================================================

def allowed_file(filename):

    if "." not in filename:
        return False

    ext = filename.rsplit(".", 1)[1].lower()

    return ext in ALLOWED_EXTENSIONS


def confidence_label(score):

    if score >= 0.90:
        return "High"

    if score >= 0.70:
        return "Medium"

    return "Low"


def json_error(message, code=400):

    return jsonify({

        "matched": False,

        "error": message

    }), code


def current_time():

    return datetime.utcnow().isoformat()


def validate_request():

    if "image" not in request.files:

        return False, json_error("No image uploaded.")

    file = request.files["image"]

    if file.filename == "":

        return False, json_error("No selected file.")

    if not allowed_file(file.filename):

        return False, json_error(
            "Unsupported file format."
        )

    return True, file

# ============================================================
# Image Processing
# ============================================================

def load_image(file):

    try:
        image_bytes = file.read()

        if not image_bytes:
            raise ValueError("Empty image file.")

        image = Image.open(io.BytesIO(image_bytes))
        image.verify()

        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")

        return image

    except (OSError, ValueError, SyntaxError) as exc:
        logger.warning("Invalid image upload: %s", exc)
        raise ValueError("Invalid image file.") from exc


def preprocess_image(image):

    """
    ResNet50V2 preprocessing
    """

    try:
        RESAMPLE = Image.Resampling.LANCZOS
    except AttributeError:
        RESAMPLE = Image.LANCZOS

    image = image.resize(

        (IMG_WIDTH, IMG_HEIGHT),

        RESAMPLE

    )

    img = np.array(image).astype(np.float32)

    img = preprocess_input(img)

    img = np.expand_dims(

        img,

        axis=0

    )

    return img


# ============================================================
# Prediction
# ============================================================

def predict(image):

    tensor = preprocess_image(image)

    prediction = MODEL.predict(

        tensor,

        verbose=0

    )[0]

    return prediction


def get_top_predictions(prediction, top=3):

    indexes = np.argsort(

        prediction

    )[::-1][:top]

    results = []

    for index in indexes:

        confidence = round(
            float(prediction[index]),
            4
        )

        results.append({

            "index": int(index),

            "localName": CLASS_NAMES[index],

            "confidence": confidence,

            "percent": round(

                confidence * 100,

                2

            )

        })

    return results


def get_herb_metadata(herb_name):

    metadata = HERB_METADATA.get(herb_name)

    if metadata is None:
        return {
            "scientificName": "",
            "family": "",
            "description": "",
        }

    return {
        "scientificName": metadata.get("scientificName", ""),
        "family": metadata.get("family", ""),
        "description": metadata.get("description", ""),
    }


def build_prediction_response(prediction):

    best_index = int(

        np.argmax(prediction)

    )

    confidence = round(
        float(prediction[best_index]),
        4
    )

    confidence_percent = round(

        confidence * 100,

        2

    )

    herb_name = CLASS_NAMES[best_index]
    herb_metadata = get_herb_metadata(herb_name)

    response = {

        "matched":

            confidence >= CONFIDENCE_THRESHOLD,

        "localName":

            herb_name,

        "confidence":

            confidence,

        "confidencePercent":

            confidence_percent,

        "confidenceLabel":

            confidence_label(confidence),

        "scientificName":

            herb_metadata["scientificName"],

        "top3":

            get_top_predictions(

                prediction,

                3

            ),

        "timestamp":

            current_time()

    }

    return response


# ============================================================
# Health Check
# ============================================================

@app.route("/")

def home():

    return jsonify({

        "status": "online",

        "project": "HerbID Iloilo",

        "model": "ResNet50V2",

        "imageSize": [

            IMG_WIDTH,

            IMG_HEIGHT

        ],

        "classes": len(CLASS_NAMES),

        "threshold": CONFIDENCE_THRESHOLD

    })


@app.route("/health")

def health():

    return jsonify({

        "status": "healthy",

        "loadedModel": MODEL is not None,

        "loadedClasses": len(CLASS_NAMES)

    })

# ============================================================
# Herb Identification Endpoint
# ============================================================

@app.route("/identify", methods=["POST"])
def identify():

    try:

        # ----------------------------------------
        # Validate request
        # ----------------------------------------

        valid, result = validate_request()

        if not valid:
            return result

        file = result

        logger.info(
            f"Prediction request received: {file.filename}"
        )

        # ----------------------------------------
        # Load image
        # ----------------------------------------

        image = load_image(file)

        # ----------------------------------------
        # Run prediction
        # ----------------------------------------

        prediction = predict(image)

        response = build_prediction_response(
            prediction
        )

        logger.info(
            f"Prediction: {response['localName']} "
            f"({response['confidencePercent']}%)"
        )

        return jsonify(response)

    except Exception as exc:

        logger.exception("Prediction failed")

        message = str(exc) if str(exc) else "Unable to process image."

        return jsonify({

            "matched": False,

            "error": message

        }), 500


# ============================================================
# 404 Handler
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "matched": False,

        "error": "Endpoint not found."

    }), 404


# ============================================================
# 405 Handler
# ============================================================

@app.errorhandler(405)
def method_not_allowed(error):

    return jsonify({

        "matched": False,

        "error": "Method not allowed."

    }), 405


# ============================================================
# File Too Large
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "matched": False,

        "error": "Image exceeds maximum size of 10 MB."

    }), 413


# ============================================================
# Bad Request
# ============================================================

@app.errorhandler(400)
def bad_request(error):

    return jsonify({

        "matched": False,

        "error": "Bad request."

    }), 400


# ============================================================
# Internal Server Error
# ============================================================

@app.errorhandler(500)
def internal_server_error(error):

    logger.exception(error)

    return jsonify({

        "matched": False,

        "error": "Internal server error."

    }), 500

# ============================================================
# Startup
# ============================================================

def startup():

    logger.info("=" * 60)
    logger.info("HerbID Iloilo API")
    logger.info("=" * 60)

    logger.info(f"Model Path : {MODEL_PATH}")
    logger.info(f"Classes    : {len(CLASS_NAMES)}")
    logger.info(f"Image Size : {IMG_WIDTH}x{IMG_HEIGHT}")
    logger.info(f"Threshold  : {CONFIDENCE_THRESHOLD}")

    try:

        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")

        if gpus:

            logger.info(f"GPU Detected ({len(gpus)})")

            for gpu in gpus:

                try:

                    tf.config.experimental.set_memory_growth(
                        gpu,
                        True
                    )

                except Exception as exc:

                    logger.warning(str(exc))

        else:

            logger.info("Running on CPU")

    except Exception as exc:

        logger.warning(str(exc))

    logger.info("=" * 60)
    logger.info("API Ready")
    logger.info("=" * 60)


# ============================================================
# Startup Hook
# ============================================================

@app.before_request
def before_request():

    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else request.remote_addr

    if request.path in {"/identify", "/health"}:
        logger.info(
            "Request from %s: %s %s",
            client_ip,
            request.method,
            request.path
        )
    else:
        logger.debug(
            "Request from %s: %s %s",
            client_ip,
            request.method,
            request.path
        )


@app.after_request
def after_request(response):

    response.headers["Cache-Control"] = "no-store"

    response.headers["X-Powered-By"] = "HerbID Iloilo"

    return response


# ============================================================
# Main
# ============================================================

startup()


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    debug = os.environ.get(
        "FLASK_DEBUG",
        "False"
    ).lower() == "true"

    app.run(

        host="0.0.0.0",

        port=port,

        debug=debug,

        threaded=True

    )