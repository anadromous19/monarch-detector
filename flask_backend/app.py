from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os
import traceback
from datetime import datetime, timezone
from decimal import Decimal
import uuid
import requests
import boto3
from werkzeug.exceptions import RequestEntityTooLarge

# =========================================================
# SETTINGS
# =========================================================

CONFIDENCE_THRESHOLD = 0.60
LOCATION_DECIMAL_PLACES = 2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.path.join(BASE_DIR, "monarch_weights.weights.h5")

WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TIMEOUT_SECONDS = 10

# Configure these as environment variables on EC2.
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
DYNAMODB_TABLE_NAME = os.environ.get(
    "DYNAMODB_TABLE_NAME",
    "MonarchSightings"
)

if not S3_BUCKET_NAME:
    raise RuntimeError("S3_BUCKET_NAME environment variable is required.")

aws_session = boto3.Session(region_name=AWS_REGION)
s3_client = aws_session.client("s3")
dynamodb = aws_session.resource("dynamodb")
sightings_table = dynamodb.Table(DYNAMODB_TABLE_NAME)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 50 * 1024 * 1024
app.config["MAX_FORM_PARTS"] = 10_000

# =========================================================
# BUILD AND LOAD CNN
# =========================================================

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Activation("sigmoid")
    ])


model = build_model()

if not os.path.exists(WEIGHTS_FILE):
    raise FileNotFoundError(f"Model weights were not found at:\n{WEIGHTS_FILE}")

model.load_weights(WEIGHTS_FILE)
print("Model loaded successfully.")
print("Weights file:", WEIGHTS_FILE)

# =========================================================
# IMAGE AND LOCATION HELPERS
# =========================================================

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def validate_coordinates(latitude_text, longitude_text):
    try:
        latitude = float(latitude_text)
        longitude = float(longitude_text)
        if -90 <= latitude <= 90 and -180 <= longitude <= 180:
            return latitude, longitude, True
    except (TypeError, ValueError):
        pass
    return None, None, False


def approximate_coordinates(latitude, longitude):
    if latitude is None or longitude is None:
        return None, None
    return (
        round(latitude, LOCATION_DECIMAL_PLACES),
        round(longitude, LOCATION_DECIMAL_PLACES)
    )

# =========================================================
# WEATHER HELPERS
# =========================================================

def empty_weather_data():
    return {
        "temperature_f": None,
        "relative_humidity_percent": None,
        "apparent_temperature_f": None,
        "weather_code": None,
        "precipitation_in": None,
        "cloud_cover_percent": None,
        "wind_speed_mph": None,
        "wind_gust_mph": None,
        "wind_direction_degrees": None,
        "surface_pressure_hpa": None,
        "visibility_miles": None,
        "solar_radiation_wm2": None,
        "is_day": None,
        "weather_time": None,
        "weather_timezone": None,
        "weather_source": None
    }


def get_current_weather(latitude, longitude):
    weather_data = empty_weather_data()
    if latitude is None or longitude is None:
        print("WEATHER SKIPPED: Valid coordinates were unavailable.")
        return weather_data

    parameters = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
            "temperature_2m", "relative_humidity_2m",
            "apparent_temperature", "weather_code", "precipitation",
            "cloud_cover", "wind_speed_10m", "wind_gusts_10m",
            "wind_direction_10m", "surface_pressure", "visibility",
            "shortwave_radiation", "is_day"
        ]),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto"
    }

    try:
        response = requests.get(
            WEATHER_API_URL,
            params=parameters,
            timeout=WEATHER_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        api_data = response.json()
        current = api_data.get("current", {})
        visibility_meters = current.get("visibility")
        visibility_miles = (
            round(float(visibility_meters) / 1609.344, 2)
            if visibility_meters is not None
            else None
        )

        return {
            "temperature_f": current.get("temperature_2m"),
            "relative_humidity_percent": current.get("relative_humidity_2m"),
            "apparent_temperature_f": current.get("apparent_temperature"),
            "weather_code": current.get("weather_code"),
            "precipitation_in": current.get("precipitation"),
            "cloud_cover_percent": current.get("cloud_cover"),
            "wind_speed_mph": current.get("wind_speed_10m"),
            "wind_gust_mph": current.get("wind_gusts_10m"),
            "wind_direction_degrees": current.get("wind_direction_10m"),
            "surface_pressure_hpa": current.get("surface_pressure"),
            "visibility_miles": visibility_miles,
            "solar_radiation_wm2": current.get("shortwave_radiation"),
            "is_day": current.get("is_day"),
            "weather_time": current.get("time"),
            "weather_timezone": api_data.get("timezone"),
            "weather_source": "Open-Meteo"
        }
    except (requests.exceptions.RequestException, TypeError, ValueError, KeyError) as error:
        print("WEATHER ERROR:", str(error))
        return weather_data

# =========================================================
# AWS STORAGE HELPERS
# =========================================================

def dynamodb_value(value):
    """DynamoDB accepts Decimal instead of Python float values."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {key: dynamodb_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [dynamodb_value(item) for item in value]
    return value


def save_sighting(
    image_bytes,
    label,
    confidence,
    latitude,
    longitude,
    timestamp,
    user_notes,
    weather_data
):
    """Save the JPEG in S3 and the sighting metadata in DynamoDB."""
    sighting_id = str(uuid.uuid4())
    image_filename = f"{sighting_id}.jpg"
    image_key = f"monarch-sightings/{timestamp[:10]}/{image_filename}"

    # Re-encode the upload as a valid JPEG before sending it to S3.
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    jpeg_buffer = io.BytesIO()
    image.save(jpeg_buffer, format="JPEG", quality=95)
    jpeg_buffer.seek(0)

    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=image_key,
        Body=jpeg_buffer.getvalue(),
        ContentType="image/jpeg",
        ServerSideEncryption="AES256"
    )

    item = {
        "sighting_id": sighting_id,
        "image_filename": image_filename,
        "image_bucket": S3_BUCKET_NAME,
        "image_key": image_key,
        "prediction": label,
        "confidence": confidence,
        "latitude": latitude,
        "longitude": longitude,
        "sighting_timestamp": timestamp,
        "year_month": timestamp[:7],
        "notes": user_notes,
        "weather": weather_data
    }

    try:
        sightings_table.put_item(Item=dynamodb_value(item))
    except Exception:
        # Avoid leaving an orphaned image when the database write fails.
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=image_key)
        raise

    return sighting_id, image_filename, image_key

# =========================================================
# ROUTES
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Monarch Flask API is running!",
        "prediction_endpoint": "/predict",
        "prediction_method": "POST",
        "weather_enabled": True,
        "storage": "Amazon S3 and DynamoDB"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("INCOMING CONTENT-LENGTH HEADER:", request.content_length)
        raw_data = request.get_data(cache=True)

        latitude_text = request.args.get("latitude", "")
        longitude_text = request.args.get("longitude", "")
        user_notes = request.args.get("notes", "")

        latitude, longitude, location_valid = validate_coordinates(
            latitude_text,
            longitude_text
        )
        latitude, longitude = approximate_coordinates(latitude, longitude)

        image_bytes = None
        if request.files:
            image_file = next(iter(request.files.values()))
            image_bytes = image_file.read()
        elif raw_data:
            image_bytes = raw_data

        if not image_bytes:
            return jsonify({"error": "No image data was received.", "saved": False}), 400

        try:
            image_array = preprocess_image(image_bytes)
        except UnidentifiedImageError:
            return jsonify({
                "error": "The uploaded data was not recognized as a valid image.",
                "saved": False
            }), 400
        except Exception as error:
            return jsonify({
                "error": f"Image processing failed: {str(error)}",
                "saved": False
            }), 400

        raw_prediction = float(model.predict(image_array, verbose=0)[0][0])
        if raw_prediction >= 0.5:
            label = "Monarch"
            confidence = raw_prediction
        else:
            label = "Non-Monarch"
            confidence = 1.0 - raw_prediction

        confidence = round(confidence, 4)
        raw_prediction = round(raw_prediction, 4)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        saved = False
        sighting_id = None
        image_filename = None
        image_key = None
        weather_data = empty_weather_data()

        if label == "Monarch" and confidence >= CONFIDENCE_THRESHOLD:
            weather_data = get_current_weather(latitude, longitude)
            sighting_id, image_filename, image_key = save_sighting(
                image_bytes=image_bytes,
                label=label,
                confidence=confidence,
                latitude=latitude,
                longitude=longitude,
                timestamp=timestamp,
                user_notes=user_notes,
                weather_data=weather_data
            )
            saved = True
            print("SIGHTING SAVED:", sighting_id)
            print("S3 IMAGE KEY:", image_key)

        response = {
            "label": label,
            "confidence": confidence,
            "raw_score": raw_prediction,
            "saved": saved,
            "sighting_id": sighting_id,
            "image_filename": image_filename,
            "latitude": latitude if location_valid else None,
            "longitude": longitude if location_valid else None,
            "location_valid": location_valid,
            "timestamp": timestamp,
            "weather": weather_data
        }
        return jsonify(response), 200

    except RequestEntityTooLarge:
        raise
    except Exception as error:
        print("UNEXPECTED SERVER ERROR:", str(error))
        traceback.print_exc()
        return jsonify({
            "error": f"Unexpected server error: {str(error)}",
            "saved": False
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Route not found.",
        "correct_prediction_route": "/predict"
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed.",
        "instructions": "Send a POST request to /predict."
    }), 405


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "The uploaded image is too large. Maximum size is 50 MB.",
        "saved": False
    }), 413


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=5000)
