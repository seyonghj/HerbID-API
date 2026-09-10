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
import base64
import time
import logging
import threading
import urllib.request
from datetime import datetime

import google.generativeai as genai
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# ============================================================
# Configuration
# ============================================================

IMG_WIDTH  = 224
IMG_HEIGHT = 224

CONFIDENCE_THRESHOLD = 0.60

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

MAX_IMAGE_SIZE = 10 * 1024 * 1024   # 10 MB

CLASS_FILE = "class_names.json"

MODEL_URL = os.environ.get("MODEL_URL", "")

MIN_MODEL_SIZE = 100 * 1024 * 1024   # 100 MB

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def resolve_model_path():
    railway_volume_dir = "/data"
    if os.path.isdir(railway_volume_dir):
        model_dir = os.path.join(railway_volume_dir, "model")
    else:
        model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    # ✅ Updated to new model filename — change this if your file has a different name
    return os.path.join(model_dir, "herb_resnet50v2.h5")


MODEL_PATH = os.environ.get("MODEL_PATH", "") or resolve_model_path()

# ============================================================
# Flask
# ============================================================

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE

genai.configure(api_key=os.environ.get("GEMINI_KEY"))

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
        raise FileNotFoundError(f"{CLASS_FILE} not found.")

    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # dict format: {"ClassName": index, ...} — sort by index value
        ordered = sorted(data.items(), key=lambda x: x[1])
        classes = [c[0] for c in ordered]
    elif isinstance(data, list):
        classes = data
    else:
        raise ValueError("Invalid class_names.json format.")

    logger.info(f"Loaded {len(classes)} herb classes: {classes}")
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

MODEL      = None
MODEL_LOCK = threading.Lock()


def is_valid_model_file(path):
    if not os.path.exists(path):
        logger.info("Model file does not exist: %s", path)
        return False

    size = os.path.getsize(path)
    if size < MIN_MODEL_SIZE:
        logger.warning(
            "Model file is too small (%.2f MB) - likely corrupt or incomplete: %s",
            size / (1024 * 1024), path
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
            path, header
        )
        return False

    logger.info("Model file verified OK (%.2f MB): %s", size / (1024 * 1024), path)
    return True


def download_model(path):
    if not MODEL_URL:
        raise RuntimeError(
            "Model file is missing/invalid and MODEL_URL is not set. "
            "Set the MODEL_URL environment variable to a direct download link."
        )

    logger.info("Downloading model from MODEL_URL to %s ...", path)
    tmp_path = path + ".part"

    try:
        with urllib.request.urlopen(MODEL_URL) as response:
            total_size  = int(response.headers.get("Content-Length", 0))
            downloaded  = 0
            chunk_size  = 1024 * 1024
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
                            logger.info(
                                "Download progress: %.2f MB / %.2f MB (%.1f%%)",
                                downloaded / (1024 * 1024),
                                total_size  / (1024 * 1024),
                                (downloaded / total_size) * 100
                            )
                        else:
                            logger.info("Download progress: %.2f MB", downloaded / (1024 * 1024))
                        last_log_time = now

        logger.info("Download complete: %.2f MB total.", downloaded / (1024 * 1024))

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
    logger.info("Checking model file at %s", MODEL_PATH)
    if is_valid_model_file(MODEL_PATH):
        logger.info("Existing model file is valid, skipping download.")
        return

    if os.path.exists(MODEL_PATH):
        logger.warning("Existing model file is invalid/corrupt. Deleting: %s", MODEL_PATH)
        try:
            os.remove(MODEL_PATH)
        except OSError as exc:
            logger.warning("Could not remove invalid model file: %s", exc)

    download_model(MODEL_PATH)

    if not is_valid_model_file(MODEL_PATH):
        raise RuntimeError(
            "Downloaded model file failed validation. Check MODEL_URL and try again."
        )


def warm_up_model(model):
    try:
        dummy = preprocess_input(np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32))
        model.predict(dummy, verbose=0)
        logger.info("Model warm-up completed.")
    except Exception as exc:
        logger.warning("Model warm-up failed: %s", exc)


def get_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL

        ensure_model_file()
        logger.info("Loading TensorFlow model into memory...")
        start = time.time()

        from tensorflow.keras.models import load_model as _load_model
        loaded = _load_model(MODEL_PATH)

        logger.info("TensorFlow model loaded in %.2f seconds.", time.time() - start)
        warm_up_model(loaded)
        MODEL = loaded
        return MODEL


# ============================================================
# Utilities
# ============================================================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def confidence_label(score):
    if score >= 0.90:
        return "High"
    if score >= 0.70:
        return "Medium"
    return "Low"


def json_error(message, code=400):
    return jsonify({"matched": False, "error": message}), code


def current_time():
    return datetime.utcnow().isoformat()


def validate_request():
    if "image" not in request.files:
        return False, json_error("No image uploaded.")
    file = request.files["image"]
    if file.filename == "":
        return False, json_error("No selected file.")
    if not allowed_file(file.filename):
        return False, json_error("Unsupported file format.")
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
    try:
        RESAMPLE = Image.Resampling.LANCZOS
    except AttributeError:
        RESAMPLE = Image.LANCZOS

    image = image.resize((IMG_WIDTH, IMG_HEIGHT), RESAMPLE)
    img   = preprocess_input(np.array(image).astype(np.float32))
    return np.expand_dims(img, axis=0)


# ============================================================
# Prediction
# ============================================================

def predict(image):
    return get_model().predict(preprocess_image(image), verbose=0)[0]


def get_top_predictions(prediction, top=3):
    indexes = np.argsort(prediction)[::-1][:top]
    return [
        {
            "index":      int(i),
            "localName":  CLASS_NAMES[i],
            "confidence": round(float(prediction[i]), 4),
            "percent":    round(float(prediction[i]) * 100, 2),
        }
        for i in indexes
    ]


def get_herb_metadata(herb_name):
    meta = HERB_METADATA.get(herb_name, {})
    return {
        "scientificName": meta.get("scientificName", ""),
        "family":         meta.get("family", ""),
        "description":    meta.get("description", ""),
    }


def build_prediction_response(prediction):
    best_index       = int(np.argmax(prediction))
    confidence       = round(float(prediction[best_index]), 4)
    confidence_pct   = round(confidence * 100, 2)
    herb_name        = CLASS_NAMES[best_index]
    meta             = get_herb_metadata(herb_name)

    return {
        "matched":          confidence >= CONFIDENCE_THRESHOLD,
        "localName":        herb_name,
        "confidence":       confidence,
        "confidencePercent": confidence_pct,
        "confidenceLabel":  confidence_label(confidence),
        "scientificName":   meta["scientificName"],
        "top3":             get_top_predictions(prediction, 3),
        "timestamp":        current_time(),
    }


# ============================================================
# Routes
# ============================================================

@app.route("/")
def home():
    return jsonify({
        "status":    "online",
        "project":   "HerbID Iloilo",
        "model":     "ResNet50V2",
        "imageSize": [IMG_WIDTH, IMG_HEIGHT],
        "classes":   len(CLASS_NAMES),
        "threshold": CONFIDENCE_THRESHOLD,
        "classNames": CLASS_NAMES,   # ✅ helpful for debugging
    })


@app.route("/health")
def health():
    return jsonify({
        "status":       "healthy",
        "loadedModel":  MODEL is not None,
        "loadedClasses": len(CLASS_NAMES),
        "classNames":   CLASS_NAMES,
    })


@app.route("/identify", methods=["POST"])
def identify():
    try:
        valid, result = validate_request()
        if not valid:
            return result

        file = result
        logger.info("Prediction request received: %s", file.filename)

        image      = load_image(file)
        prediction = predict(image)
        response   = build_prediction_response(prediction)

        logger.info(
            "Prediction: %s (%.2f%%)",
            response["localName"],
            response["confidencePercent"]
        )

        return jsonify(response)

    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({
            "matched": False,
            "error":   str(exc) or "Unable to process image."
        }), 500


@app.route("/gemini-verify", methods=["POST"])
def gemini_verify():
    try:
        data       = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        image_b64  = data.get("image")
        mime_type  = data.get("mimeType", "image/jpeg")
        prompt     = data.get("prompt")

        if not image_b64 or not prompt:
            return jsonify({"error": "Missing image or prompt"}), 400

        model    = genai.GenerativeModel("gemini-3.6-flash")
        response = model.generate_content([
            {"mime_type": mime_type, "data": image_b64},
            prompt
        ])

        text  = response.text
        clean = text.replace("```json", "").replace("```", "").strip()
        return jsonify(json.loads(clean))

    except json.JSONDecodeError as exc:
        logger.warning("Gemini returned non-JSON: %s", exc)
        return jsonify({"error": "Gemini returned invalid JSON", "raw": text}), 500
    except Exception as exc:
        logger.exception("Gemini verify failed")
        return jsonify({"error": str(exc)}), 500


# ============================================================
# Error Handlers
# ============================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"matched": False, "error": "Endpoint not found."}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"matched": False, "error": "Method not allowed."}), 405

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"matched": False, "error": "Image exceeds maximum size of 10 MB."}), 413

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"matched": False, "error": "Bad request."}), 400

@app.errorhandler(500)
def internal_server_error(e):
    logger.exception(e)
    return jsonify({"matched": False, "error": "Internal server error."}), 500


# ============================================================
# Request Hooks
# ============================================================

@app.before_request
def before_request():
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip     = forwarded_for.split(",")[0].strip() if forwarded_for else request.remote_addr
    if request.path in {"/identify", "/health", "/gemini-verify"}:
        logger.info("Request from %s: %s %s", client_ip, request.method, request.path)
    else:
        logger.debug("Request from %s: %s %s", client_ip, request.method, request.path)


@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Powered-By"]  = "HerbID Iloilo"
    return response


# ============================================================
# Startup
# ============================================================

def startup():
    logger.info("=" * 60)
    logger.info("HerbID Iloilo API")
    logger.info("=" * 60)
    logger.info("Model Path  : %s", MODEL_PATH)
    logger.info("Model URL   : %s", "set" if MODEL_URL else "NOT SET")
    logger.info("Classes     : %d → %s", len(CLASS_NAMES), CLASS_NAMES)
    logger.info("Image Size  : %dx%d", IMG_WIDTH, IMG_HEIGHT)
    logger.info("Threshold   : %s", CONFIDENCE_THRESHOLD)
    logger.info("Model will be loaded lazily on first /identify request.")

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info("GPU Detected (%d)", len(gpus))
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as exc:
                    logger.warning(str(exc))
        else:
            logger.info("Running on CPU")
    except Exception as exc:
        logger.warning(str(exc))

    logger.info("=" * 60)
    logger.info("API Ready")
    logger.info("=" * 60)


startup()

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)