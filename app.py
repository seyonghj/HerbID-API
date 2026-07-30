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
import time
import logging
import threading
import urllib.request
from datetime import datetime

import numpy as np

from PIL import Image

from flask import (
    Flask,
    request,
    jsonify
)

from flask_cors import CORS

from tensorflow.keras.applications.resnet_v2 import preprocess_input

# ============================================================
# Configuration
# ============================================================

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

CLASS_FILE = "class_names.json"

# Model download URL (configurable via environment variable)
MODEL_URL = os.environ.get("MODEL_URL", "")

# Minimum plausible size for a valid HDF5 Keras model (in bytes).
# Real HerbID model is expected to be well over 100 MB.
MIN_MODEL_SIZE = 100 * 1024 * 1024   # 100 MB

# HDF5 file signature (first 8 bytes of any valid .h5 file)
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def resolve_model_path():
    """
    Railway-compatible model path resolution.

    - If a Railway Volume is mounted at /data, store/load the model there
      so it survives redeploys.
    - Otherwise fall back to the local ./model directory for local dev,
      Docker, Render, etc.
    """
    railway_volume_dir = "/data"

    if os.path.isdir(railway_volume_dir):
        model_dir = os.path.join(railway_volume_dir, "model")
    else:
        model_dir = "model"

    os.makedirs(model_dir, exist_ok=True)

    return os.path.join(model_dir, "herb_resnet50v2.h5")


MODEL_PATH = os.environ.get("MODEL_PATH", "") or resolve_model_path()

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
# Lazy, Thread-Safe Model Loading
# ============================================================
#
# The model is NOT loaded at import time. This keeps Flask import
# fast and crash-free even if the model file is missing or the
# download source is temporarily unreachable. The model is loaded
# on first use (first /identify request, or first call to
# get_model()), then cached in memory for the lifetime of the
# process.
# ============================================================

MODEL = None
MODEL_LOCK = threading.Lock()


def is_valid_model_file(path):
    """
    Verify a model file on disk actually looks like a usable
    Keras/HDF5 model before we try to load it with TensorFlow.
    """

    if not os.path.exists(path):
        logger.info("Model file does not exist: %s", path)
        return False

    size = os.path.getsize(path)

    if size < MIN_MODEL_SIZE:
        logger.warning(
            "Model file is too small (%.2f MB) - likely corrupt or incomplete: %s",
            size / (1024 * 1024),
            path
        )
        return False

    try:
        with open(path, "rb") as f:
            header = f.read(8)
    except OSError as exc:
        logger.warning("Could not read model file header: %s", exc)
        return False

    if header != HDF5_SIGNATURE:
        logger.warning(
            "Model file does not have a valid HDF5 signature: %s (header=%r)",
            path,
            header
        )
        return False

    logger.info(
        "Model file verified OK (%.2f MB): %s",
        size / (1024 * 1024),
        path
    )

    return True


def download_model(path):
    """
    Download the model file from MODEL_URL to `path`, streaming to
    disk with periodic progress logging. Downloads to a temporary
    file first and only renames to the final path on success, so a
    failed/partial download never leaves a bad file at `path`.
    """

    if not MODEL_URL:
        raise RuntimeError(
            "Model file is missing/invalid and MODEL_URL is not set. "
            "Set the MODEL_URL environment variable to a direct "
            "download link for herb_resnet50v2.h5."
        )

    logger.info("Downloading model from MODEL_URL to %s ...", path)

    tmp_path = path + ".part"

    try:
        with urllib.request.urlopen(MODEL_URL) as response:

            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB
            last_log_time = time.time()

            with open(tmp_path, "wb") as out_file:

                while True:

                    chunk = response.read(chunk_size)

                    if not chunk:
                        break

                    out_file.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()

                    if now - last_log_time >= 2:

                        if total_size:
                            percent = (downloaded / total_size) * 100
                            logger.info(
                                "Download progress: %.2f MB / %.2f MB (%.1f%%)",
                                downloaded / (1024 * 1024),
                                total_size / (1024 * 1024),
                                percent
                            )
                        else:
                            logger.info(
                                "Download progress: %.2f MB",
                                downloaded / (1024 * 1024)
                            )

                        last_log_time = now

        logger.info(
            "Download complete: %.2f MB total.",
            downloaded / (1024 * 1024)
        )

    except Exception as exc:
        logger.exception("Model download failed: %s", exc)

        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        raise

    os.replace(tmp_path, path)


def ensure_model_file():
    """
    Make sure a valid model file exists at MODEL_PATH, downloading
    (or re-downloading) it if necessary.
    """

    logger.info("Checking model file at %s", MODEL_PATH)

    if is_valid_model_file(MODEL_PATH):
        logger.info("Existing model file is valid, skipping download.")
        return

    if os.path.exists(MODEL_PATH):
        logger.warning(
            "Existing model file is invalid/corrupt. Deleting: %s",
            MODEL_PATH
        )
        try:
            os.remove(MODEL_PATH)
        except OSError as exc:
            logger.warning("Could not remove invalid model file: %s", exc)

    download_model(MODEL_PATH)

    if not is_valid_model_file(MODEL_PATH):
        raise RuntimeError(
            "Downloaded model file failed validation. "
            "Check MODEL_URL and try again."
        )


def warm_up_model(model):

    try:
        dummy_input = np.zeros(
            (1, IMG_HEIGHT, IMG_WIDTH, 3),
            dtype=np.float32
        )
        dummy_input = preprocess_input(dummy_input)
        model.predict(dummy_input, verbose=0)
        logger.info("Model warm-up completed.")
    except Exception as exc:
        logger.warning("Model warm-up failed: %s", exc)


def get_model():
    """
    Thread-safe singleton accessor for the TensorFlow model.

    - First caller triggers verification/download (if needed) and
      the actual TensorFlow load.
    - All subsequent callers (including concurrent requests) reuse
      the already-loaded model instance.
    - TensorFlow itself is only imported here, on first use, so
      Flask import/startup never depends on TensorFlow being ready.
    """

    global MODEL

    if MODEL is not None:
        return MODEL

    with MODEL_LOCK:

        # Re-check in case another thread loaded it while we waited
        # for the lock.
        if MODEL is not None:
            return MODEL

        ensure_model_file()

        logger.info("Loading TensorFlow model into memory...")

        start_time = time.time()

        # Imported lazily so a missing/broken TensorFlow install (or
        # slow import) never blocks Flask from starting up and
        # serving /  and /health.
        from tensorflow.keras.models import load_model as _load_model

        loaded_model = _load_model(MODEL_PATH)

        elapsed = time.time() - start_time

        logger.info(
            "TensorFlow model loaded successfully in %.2f seconds.",
            elapsed
        )

        warm_up_model(loaded_model)

        MODEL = loaded_model

        return MODEL


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

    model = get_model()

    tensor = preprocess_image(image)

    prediction = model.predict(

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
    logger.info(f"Model URL  : {'set' if MODEL_URL else 'NOT SET'}")
    logger.info(f"Classes    : {len(CLASS_NAMES)}")
    logger.info(f"Image Size : {IMG_WIDTH}x{IMG_HEIGHT}")
    logger.info(f"Threshold  : {CONFIDENCE_THRESHOLD}")
    logger.info("Model will be loaded lazily on first /identify request.")

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
