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

import google.generativeai as genai
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# ============================================================
# Configuration
# ============================================================

IMG_WIDTH  = 224
IMG_HEIGHT = 224

# ✅ Lowered threshold slightly — ResNet50V2 softmax scores
# are often compressed; 0.55 catches more correct matches
# that were previously rejected just below 0.60.
CONFIDENCE_THRESHOLD = 0.55

# Top-N predictions to return
TOP_N = 5

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "heic", "heif"}

MAX_IMAGE_SIZE = 15 * 1024 * 1024   # 15 MB

CLASS_FILE = "class_names.json"

MODEL_URL = os.environ.get("MODEL_URL", "")

MIN_MODEL_SIZE = 50 * 1024 * 1024   # 50 MB

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"

# ============================================================
# Preprocessing constants
# ============================================================

# Test-time augmentation: run N slightly varied versions of the
# image and average the predictions. More = more accurate but slower.
# 1 = disabled (fastest), 5 = good balance, 9 = slowest/most accurate
TTA_STEPS = 5

# ============================================================
# Model path
# ============================================================

def resolve_model_path():
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

# ── Gemini key rotation ───────────────────────────────────
# Collect all keys from env: GEMINI_KEY, GEMINI_KEY_1, GEMINI_KEY_2, ...
# If one key hits its daily quota (429 / Resource Exhausted), the route
# automatically tries the next key in the list.
_GEMINI_KEYS = [
    k for k in [
        os.environ.get("GEMINI_KEY"),
        os.environ.get("GEMINI_KEY_1"),
        os.environ.get("GEMINI_KEY_2"),
        os.environ.get("GEMINI_KEY_3"),
        os.environ.get("GEMINI_KEY_4"),
        os.environ.get("GEMINI_KEY_5"),
    ] if k
]
if not _GEMINI_KEYS:
    logger.warning("No Gemini API keys found in environment variables.")

# Configure with the first key by default (used by non-rotating code paths)
if _GEMINI_KEYS:
    genai.configure(api_key=_GEMINI_KEYS[0])

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
    herb_name: {"scientificName": "", "family": "", "description": ""}
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
            "Model file too small (%.2f MB): %s",
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
        logger.warning("Invalid HDF5 signature: %s", path)
        return False

    logger.info("Model file OK (%.2f MB): %s", size / (1024 * 1024), path)
    return True


def download_model(path):
    if not MODEL_URL:
        raise RuntimeError(
            "Model file missing and MODEL_URL not set. "
            "Set MODEL_URL environment variable."
        )

    logger.info("Downloading model to %s ...", path)
    tmp_path = path + ".part"

    try:
        with urllib.request.urlopen(MODEL_URL) as response:
            total_size    = int(response.headers.get("Content-Length", 0))
            downloaded    = 0
            chunk_size    = 1024 * 1024
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
                        pct = f"{(downloaded/total_size)*100:.1f}%" if total_size else ""
                        logger.info(
                            "Download: %.2f MB %s",
                            downloaded / (1024 * 1024), pct
                        )
                        last_log_time = now

        logger.info("Download complete: %.2f MB", downloaded / (1024 * 1024))
    except Exception as exc:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    os.replace(tmp_path, path)


def ensure_model_file():
    logger.info("Checking model at %s", MODEL_PATH)
    if is_valid_model_file(MODEL_PATH):
        return

    if os.path.exists(MODEL_PATH):
        try:
            os.remove(MODEL_PATH)
        except OSError:
            pass

    download_model(MODEL_PATH)

    if not is_valid_model_file(MODEL_PATH):
        raise RuntimeError("Downloaded model failed validation.")


def warm_up_model(model):
    try:
        dummy = preprocess_input(
            np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        )
        model.predict(dummy, verbose=0)
        logger.info("Model warm-up done.")
    except Exception as exc:
        logger.warning("Warm-up failed: %s", exc)


def get_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL

        ensure_model_file()
        logger.info("Loading TensorFlow model...")
        start = time.time()

        from tensorflow.keras.models import load_model as _load_model
        loaded = _load_model(MODEL_PATH)

        logger.info("Model loaded in %.2fs", time.time() - start)
        warm_up_model(loaded)
        MODEL = loaded
        return MODEL


# ============================================================
# ✅ IMPROVED IMAGE PREPROCESSING
# ============================================================

def smart_resize(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Resize with aspect-ratio-preserving crop (center crop).
    This is better than plain stretch-resize because it avoids
    distorting leaves and stems that the model learned to recognize.

    Pipeline:
    1. Convert to RGB (handles RGBA, grayscale, CMYK, palette modes)
    2. Scale so the shortest side = target size (keeps aspect ratio)
    3. Center-crop to exactly target_w × target_h
    """
    image = image.convert("RGB")

    orig_w, orig_h = image.size

    # Scale so shortest side = target
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w = max(int(round(orig_w * scale)), target_w)
    new_h = max(int(round(orig_h * scale)), target_h)

    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS

    image = image.resize((new_w, new_h), resample)

    # Center crop
    left   = (new_w - target_w) // 2
    top    = (new_h - target_h) // 2
    image  = image.crop((left, top, left + target_w, top + target_h))

    return image


def normalize_image(image: Image.Image) -> Image.Image:
    """
    Gentle normalization to handle common real-world photo issues:
    - Auto-levels (removes extreme dark/bright bias)
    - Slight sharpening (compensates for phone camera softness)
    Does NOT aggressively alter color since ResNet50V2 preprocess_input
    handles the actual normalization to [-1, 1].
    """
    # Auto-equalize levels gently using autocontrast
    image = ImageOps.autocontrast(image, cutoff=1)

    # Slight sharpening — helps with blurry phone photos
    enhancer = ImageEnhance.Sharpness(image)
    image    = enhancer.enhance(1.3)

    return image


def image_to_tensor(image: Image.Image) -> np.ndarray:
    """Convert PIL image to preprocessed numpy tensor for ResNet50V2."""
    arr = np.array(image, dtype=np.float32)
    arr = preprocess_input(arr)          # scales to [-1, 1] for ResNet50V2
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Full preprocessing pipeline: normalize → resize → tensorize."""
    image = normalize_image(image)
    image = smart_resize(image, IMG_WIDTH, IMG_HEIGHT)
    return image_to_tensor(image)


# ============================================================
# ✅ TEST-TIME AUGMENTATION (TTA)
# ============================================================

def tta_variants(image: Image.Image):
    """
    Generate slightly varied versions of the same image for TTA.
    Averaging predictions over multiple crops/flips reduces the
    chance of a bad result caused by framing or orientation.

    Variants:
    - Center crop (same as standard)
    - Horizontal flip
    - Slight random crops from 4 corners (top-left, top-right, bottom-left, bottom-right)
    """
    image = image.convert("RGB")
    orig_w, orig_h = image.size

    # Scale to slightly larger than target first
    pad    = 14  # 6% larger than 224
    target = IMG_WIDTH + pad * 2

    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS

    scale  = max(target / orig_w, target / orig_h)
    new_w  = max(int(round(orig_w * scale)), target)
    new_h  = max(int(round(orig_h * scale)), target)
    scaled = image.resize((new_w, new_h), resample)

    def crop(img, x, y):
        return img.crop((x, y, x + IMG_WIDTH, y + IMG_HEIGHT))

    cx = (new_w - IMG_WIDTH) // 2
    cy = (new_h - IMG_HEIGHT) // 2

    variants = [
        crop(scaled, cx,          cy),           # center
        crop(scaled, cx,          cy).transpose(Image.FLIP_LEFT_RIGHT),  # flip
        crop(scaled, 0,           0),             # top-left
        crop(scaled, new_w - IMG_WIDTH, 0),       # top-right
        crop(scaled, 0,           new_h - IMG_HEIGHT),   # bottom-left
        crop(scaled, new_w - IMG_WIDTH, new_h - IMG_HEIGHT),  # bottom-right
        crop(scaled, cx,          0),             # top-center
        crop(scaled, 0,           cy),            # left-center
        crop(scaled, new_w - IMG_WIDTH, cy),      # right-center
    ]

    return variants[:TTA_STEPS]


def predict_with_tta(image: Image.Image) -> np.ndarray:
    """
    Run prediction with test-time augmentation.
    Returns averaged softmax probabilities over all TTA variants.
    """
    model    = get_model()
    variants = tta_variants(image)

    # Normalize each variant then stack into a batch
    tensors = []
    for v in variants:
        v = normalize_image(v)
        tensors.append(image_to_tensor(v)[0])  # remove batch dim

    batch       = np.stack(tensors, axis=0)   # (TTA_STEPS, H, W, 3)
    predictions = model.predict(batch, verbose=0)   # (TTA_STEPS, num_classes)
    averaged    = predictions.mean(axis=0)     # (num_classes,)

    return averaged


# ============================================================
# Utilities
# ============================================================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def confidence_label(score):
    if score >= 0.85:
        return "High"
    if score >= 0.65:
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
# Image Loading
# ============================================================

def load_image(file) -> Image.Image:
    """
    Load and validate uploaded image.
    Handles HEIC/HEIF via pillow-heif if installed,
    and fixes EXIF orientation automatically.
    """
    try:
        image_bytes = file.read()
        if not image_bytes:
            raise ValueError("Empty image file.")

        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
        image = Image.open(io.BytesIO(image_bytes))

        # ✅ Fix EXIF rotation (phone photos often come in sideways)
        image = ImageOps.exif_transpose(image)

        # Convert to RGB
        image = image.convert("RGB")

        return image

    except (OSError, ValueError, SyntaxError) as exc:
        logger.warning("Invalid image: %s", exc)
        raise ValueError("Invalid image file.") from exc


# ============================================================
# Prediction helpers
# ============================================================

def get_top_predictions(prediction, top=TOP_N):
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
    best_index     = int(np.argmax(prediction))
    confidence     = round(float(prediction[best_index]), 4)
    confidence_pct = round(confidence * 100, 2)
    herb_name      = CLASS_NAMES[best_index]
    meta           = get_herb_metadata(herb_name)

    # ✅ Entropy-based uncertainty check:
    # If the model is spreading confidence evenly across many classes
    # (high entropy), treat it as no match even if one class is
    # technically the highest.
    probs       = prediction.astype(np.float64)
    probs       = np.clip(probs, 1e-9, 1.0)
    entropy     = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(CLASS_NAMES))
    norm_entropy = entropy / max_entropy   # 0 = certain, 1 = random

    # If entropy is very high (model is confused) treat as no match
    uncertain = norm_entropy > 0.75

    return {
        "matched":           (confidence >= CONFIDENCE_THRESHOLD) and not uncertain,
        "localName":         herb_name,
        "confidence":        confidence,
        "confidencePercent": confidence_pct,
        "confidenceLabel":   confidence_label(confidence),
        "scientificName":    meta["scientificName"],
        "top3":              get_top_predictions(prediction, 3),
        "top5":              get_top_predictions(prediction, 5),
        "entropy":           round(float(norm_entropy), 4),
        "uncertain":         uncertain,
        "timestamp":         current_time(),
    }


# ============================================================
# Routes
# ============================================================

@app.route("/")
def home():
    return jsonify({
        "status":     "online",
        "project":    "HerbID Iloilo",
        "model":      "ResNet50V2",
        "imageSize":  [IMG_WIDTH, IMG_HEIGHT],
        "classes":    len(CLASS_NAMES),
        "threshold":  CONFIDENCE_THRESHOLD,
        "ttaSteps":   TTA_STEPS,
        "classNames": CLASS_NAMES,
    })


@app.route("/health")
def health():
    return jsonify({
        "status":        "healthy",
        "loadedModel":   MODEL is not None,
        "loadedClasses": len(CLASS_NAMES),
        "classNames":    CLASS_NAMES,
        "ttaSteps":      TTA_STEPS,
    })


@app.route("/identify", methods=["POST"])
def identify():
    try:
        valid, result = validate_request()
        if not valid:
            return result

        file = result
        logger.info("Identify request: %s", file.filename)

        # Load + preprocess
        image      = load_image(file)
        prediction = predict_with_tta(image)
        response   = build_prediction_response(prediction)

        logger.info(
            "Result: %s (%.2f%%) entropy=%.3f matched=%s",
            response["localName"],
            response["confidencePercent"],
            response["entropy"],
            response["matched"],
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
    if not _GEMINI_KEYS:
        return jsonify({"error": "No Gemini API keys configured on the server."}), 500

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    image_b64 = data.get("image")
    mime_type = data.get("mimeType", "image/jpeg")
    prompt    = data.get("prompt")

    if not image_b64 or not prompt:
        return jsonify({"error": "Missing image or prompt"}), 400

    last_error = None
    text       = None

    # Try each key in order — stops as soon as one succeeds.
    # Skips to the next key on quota errors (429 / ResourceExhausted).
    for idx, api_key in enumerate(_GEMINI_KEYS):
        try:
            genai.configure(api_key=api_key)
            model    = genai.GenerativeModel("gemini-3.6-flash")
            response = model.generate_content([
                {"mime_type": mime_type, "data": image_b64},
                prompt
            ])
            text  = response.text
            clean = text.replace("```json", "").replace("```", "").strip()

            try:
                result = json.loads(clean)
            except json.JSONDecodeError:
                logger.warning("Gemini key %d returned non-JSON: %.200s", idx + 1, text)
                return jsonify({"error": "Gemini returned invalid JSON", "raw": text}), 500

            if idx > 0:
                logger.info("Gemini succeeded on key %d (keys 1–%d exhausted)", idx + 1, idx)
            return jsonify(result)

        except Exception as exc:
            err_str = str(exc).lower()
            # 429 / quota exhausted / resource exhausted → try next key
            if any(kw in err_str for kw in ["429", "quota", "resource exhausted", "rate limit"]):
                logger.warning(
                    "Gemini key %d quota hit (%s) — trying next key", idx + 1, str(exc)[:120]
                )
                last_error = exc
                continue
            # Any other error (bad key, network, model error) → fail immediately
            logger.exception("Gemini key %d failed with non-quota error", idx + 1)
            return jsonify({"error": str(exc)}), 500

    # All keys exhausted
    logger.error("All %d Gemini keys quota-exhausted. Last error: %s", len(_GEMINI_KEYS), last_error)
    return jsonify({
        "error": f"All {len(_GEMINI_KEYS)} Gemini API key(s) have reached their daily quota. Please try again after midnight Pacific Time.",
        "quota_exhausted": True,
    }), 429


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
    return jsonify({"matched": False, "error": "Image exceeds 15 MB limit."}), 413

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
    logger.info("Classes     : %d", len(CLASS_NAMES))
    logger.info("Image Size  : %dx%d", IMG_WIDTH, IMG_HEIGHT)
    logger.info("Threshold   : %s", CONFIDENCE_THRESHOLD)
    logger.info("TTA Steps   : %d", TTA_STEPS)
    logger.info("Class Names : %s", CLASS_NAMES)

    try:
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
