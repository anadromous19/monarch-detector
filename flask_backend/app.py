from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os
import csv
import traceback
from datetime import datetime
import uuid
import requests
from werkzeug.exceptions import RequestEntityTooLarge

# =========================================================
# SETTINGS
# =========================================================

CONFIDENCE_THRESHOLD = 0.60

# Folder containing this Python file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model weights must be in the same folder as this Python file
WEIGHTS_FILE = os.path.join(
    BASE_DIR,
    "monarch_weights.weights.h5"
)

# Folder where confident monarch images will be saved
SAVE_FOLDER = os.path.join(BASE_DIR, "monarch_saved_images")

# CSV file where confident monarch sightings will be recorded
CSV_FILE = os.path.join(BASE_DIR, "sightings.csv")

# Open-Meteo weather endpoint
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TIMEOUT_SECONDS = 10

os.makedirs(SAVE_FOLDER, exist_ok=True)

app = Flask(__name__)
# Maximum uploaded image size: 50 MB
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# Newer Werkzeug versions (2.3+) added separate caps for multipart
# form parsing that did not exist before. These can reject a normal
# single-image upload even though MAX_CONTENT_LENGTH is plenty large,
# if a newer Werkzeug ends up installed (e.g. after a fresh AWS
# deployment) than whatever was cached locally during development.
app.config["MAX_FORM_MEMORY_SIZE"] = 50 * 1024 * 1024  # default is 500 KB
app.config["MAX_FORM_PARTS"] = 10_000                   # default is 1000

# =========================================================
# BUILD AND LOAD CNN
# =========================================================

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            padding="same"
        ),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            padding="same"
        ),
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
    raise FileNotFoundError(
        f"Model weights were not found at:\n{WEIGHTS_FILE}"
    )

model.load_weights(WEIGHTS_FILE)

print("Model loaded successfully.")
print("Weights file:", WEIGHTS_FILE)


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image_bytes):
    """
    Convert image bytes into a normalized NumPy array.

    Final shape:
        (1, 224, 224, 3)
    """

    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =========================================================
# COORDINATE VALIDATION
# =========================================================

def validate_coordinates(latitude_text, longitude_text):
    """
    Convert coordinate text into numbers and validate their ranges.

    Returns:
        latitude
        longitude
        location_valid
    """

    try:
        latitude = float(latitude_text)
        longitude = float(longitude_text)

        latitude_valid = -90 <= latitude <= 90
        longitude_valid = -180 <= longitude <= 180

        if latitude_valid and longitude_valid:
            return latitude, longitude, True

    except (TypeError, ValueError):
        pass

    return None, None, False


# =========================================================
# WEATHER HELPERS
# =========================================================

def empty_weather_data():
    """
    Return blank weather values if coordinates are unavailable
    or if the weather request fails.
    """

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
    """
    Fetch current weather from Open-Meteo using the sighting's
    latitude and longitude.

    A weather failure does not stop image prediction or saving.
    Blank weather values are returned instead.
    """

    weather_data = empty_weather_data()

    if latitude is None or longitude is None:
        print("WEATHER SKIPPED: Valid coordinates were unavailable.")
        return weather_data

    parameters = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "weather_code",
            "precipitation",
            "cloud_cover",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m",
            "surface_pressure",
            "visibility",
            "shortwave_radiation",
            "is_day"
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

        if visibility_meters is not None:
            visibility_miles = round(
                float(visibility_meters) / 1609.344,
                2
            )
        else:
            visibility_miles = None

        weather_data = {
            "temperature_f": current.get("temperature_2m"),
            "relative_humidity_percent": current.get(
                "relative_humidity_2m"
            ),
            "apparent_temperature_f": current.get(
                "apparent_temperature"
            ),
            "weather_code": current.get("weather_code"),
            "precipitation_in": current.get("precipitation"),
            "cloud_cover_percent": current.get("cloud_cover"),
            "wind_speed_mph": current.get("wind_speed_10m"),
            "wind_gust_mph": current.get("wind_gusts_10m"),
            "wind_direction_degrees": current.get(
                "wind_direction_10m"
            ),
            "surface_pressure_hpa": current.get(
                "surface_pressure"
            ),
            "visibility_miles": visibility_miles,
            "solar_radiation_wm2": current.get(
                "shortwave_radiation"
            ),
            "is_day": current.get("is_day"),
            "weather_time": current.get("time"),
            "weather_timezone": api_data.get("timezone"),
            "weather_source": "Open-Meteo"
        }

        print("WEATHER RETRIEVED SUCCESSFULLY")
        print("WEATHER DATA:", weather_data)

        return weather_data

    except requests.exceptions.Timeout:
        print("WEATHER ERROR: Open-Meteo request timed out.")
        return weather_data

    except requests.exceptions.RequestException as error:
        print("WEATHER REQUEST ERROR:", str(error))
        return weather_data

    except (TypeError, ValueError, KeyError) as error:
        print("WEATHER DATA ERROR:", str(error))
        return weather_data


def csv_value(value):
    """
    Preserve legitimate zero values in the CSV while converting
    missing None values to blank cells.
    """

    return "" if value is None else value


# =========================================================
# SAVE A MONARCH SIGHTING
# =========================================================

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
    """
    Save the image and append one sighting row to the CSV.

    location_valid is intentionally not stored in the CSV.
    """

    sighting_id = str(uuid.uuid4())
    image_filename = f"{sighting_id}.jpg"

    saved_image_path = os.path.join(
        SAVE_FOLDER,
        image_filename
    )

    # Re-save through Pillow to ensure the file is a valid JPEG
    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image.save(
        saved_image_path,
        format="JPEG",
        quality=95
    )

    csv_already_exists = os.path.exists(CSV_FILE)

    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.writer(csvfile)

        if not csv_already_exists:
            writer.writerow([
                "sighting_id",
                "image_filename",
                "prediction",
                "confidence",
                "latitude",
                "longitude",
                "sighting_timestamp",
                "notes",
                "temperature_f",
                "relative_humidity_percent",
                "apparent_temperature_f",
                "weather_code",
                "precipitation_in",
                "cloud_cover_percent",
                "wind_speed_mph",
                "wind_gust_mph",
                "wind_direction_degrees",
                "surface_pressure_hpa",
                "visibility_miles",
                "solar_radiation_wm2",
                "is_day",
                "weather_time",
                "weather_timezone",
                "weather_source"
            ])

        writer.writerow([
            sighting_id,
            image_filename,
            label,
            confidence,
            csv_value(latitude),
            csv_value(longitude),
            timestamp,
            user_notes,
            csv_value(weather_data.get("temperature_f")),
            csv_value(
                weather_data.get("relative_humidity_percent")
            ),
            csv_value(
                weather_data.get("apparent_temperature_f")
            ),
            csv_value(weather_data.get("weather_code")),
            csv_value(weather_data.get("precipitation_in")),
            csv_value(
                weather_data.get("cloud_cover_percent")
            ),
            csv_value(weather_data.get("wind_speed_mph")),
            csv_value(weather_data.get("wind_gust_mph")),
            csv_value(
                weather_data.get("wind_direction_degrees")
            ),
            csv_value(
                weather_data.get("surface_pressure_hpa")
            ),
            csv_value(weather_data.get("visibility_miles")),
            csv_value(
                weather_data.get("solar_radiation_wm2")
            ),
            csv_value(weather_data.get("is_day")),
            csv_value(weather_data.get("weather_time")),
            csv_value(weather_data.get("weather_timezone")),
            csv_value(weather_data.get("weather_source"))
        ])

    return sighting_id, image_filename, saved_image_path


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Monarch Flask API is running!",
        "prediction_endpoint": "/predict",
        "prediction_method": "POST",
        "weather_enabled": True,
        "weather_source": "Open-Meteo"
    }), 200


# =========================================================
# PREDICTION ROUTE
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -------------------------------------------------
        # 1. Read raw request data first
        # -------------------------------------------------

        print(
            "INCOMING CONTENT-LENGTH HEADER:",
            request.content_length,
            "CONTENT-TYPE:",
            request.content_type
        )

        raw_data = request.get_data(cache=True)

        # -------------------------------------------------
        # 2. Read latitude, longitude, and notes from URL
        # -------------------------------------------------

        latitude_text = request.args.get(
            "latitude",
            ""
        )

        longitude_text = request.args.get(
            "longitude",
            ""
        )

        user_notes = request.args.get(
            "notes",
            ""
        )

        latitude, longitude, location_valid = validate_coordinates(
            latitude_text,
            longitude_text
        )

        print()
        print("========================================")
        print("NEW PREDICTION REQUEST")
        print("URL:", request.url)
        print("METHOD:", request.method)
        print("CONTENT TYPE:", request.content_type)
        print("CONTENT LENGTH:", request.content_length)
        print("RAW DATA SIZE:", len(raw_data))
        print("FIRST 20 BYTES:", raw_data[:20])
        print("LATITUDE RECEIVED:", latitude_text)
        print("LONGITUDE RECEIVED:", longitude_text)
        print("LOCATION VALID:", location_valid)
        print("========================================")

        # -------------------------------------------------
        # 3. Obtain uploaded image bytes
        # -------------------------------------------------

        image_bytes = None
        upload_method = None

        # Normal multipart upload
        if request.files:
            image_file = next(
                iter(request.files.values())
            )

            image_bytes = image_file.read()
            upload_method = "multipart file upload"

        # MIT App Inventor Web.PostFile raw body
        elif raw_data:
            image_bytes = raw_data
            upload_method = "raw POST body"

        if not image_bytes:
            return jsonify({
                "error": "No image data was received.",
                "content_type": request.content_type,
                "raw_data_size": len(raw_data),
                "latitude": latitude_text,
                "longitude": longitude_text,
                "saved": False
            }), 400

        print("UPLOAD METHOD:", upload_method)
        print("IMAGE BYTE SIZE:", len(image_bytes))

        # -------------------------------------------------
        # 4. Validate and preprocess image
        # -------------------------------------------------

        try:
            image_array = preprocess_image(
                image_bytes
            )

            print("IMAGE PROCESSED SUCCESSFULLY")
            print(
                "MODEL INPUT SHAPE:",
                image_array.shape
            )

        except UnidentifiedImageError:
            print(
                "IMAGE ERROR: Pillow could not identify "
                "the uploaded data."
            )

            print(
                "FIRST 100 BYTES:",
                image_bytes[:100]
            )

            return jsonify({
                "error": (
                    "The server received data, but it was "
                    "not recognized as a valid image."
                ),
                "content_type": request.content_type,
                "image_byte_size": len(image_bytes),
                "first_bytes": repr(
                    image_bytes[:40]
                ),
                "saved": False
            }), 400

        except Exception as error:
            print(
                "IMAGE PROCESSING ERROR:",
                str(error)
            )

            traceback.print_exc()

            return jsonify({
                "error": (
                    f"Image processing failed: {str(error)}"
                ),
                "content_type": request.content_type,
                "image_byte_size": len(image_bytes),
                "saved": False
            }), 400

        # -------------------------------------------------
        # 5. Run CNN prediction
        # -------------------------------------------------

        raw_prediction = float(
            model.predict(
                image_array,
                verbose=0
            )[0][0]
        )

        if raw_prediction >= 0.5:
            label = "Monarch"
            confidence = raw_prediction
        else:
            label = "Non-Monarch"
            confidence = 1.0 - raw_prediction

        confidence = round(
            confidence,
            4
        )

        raw_prediction = round(
            raw_prediction,
            4
        )

        print("PREDICTION:", label)
        print("CONFIDENCE:", confidence)
        print("RAW SCORE:", raw_prediction)

        # -------------------------------------------------
        # 6. Prepare response and weather values
        # -------------------------------------------------

        saved = False
        sighting_id = None
        image_filename = None
        weather_data = empty_weather_data()

        timestamp = datetime.now().isoformat(
            timespec="seconds"
        )

        # -------------------------------------------------
        # 7. Save confident Monarch sightings
        # -------------------------------------------------

        if (
            label == "Monarch"
            and confidence >= CONFIDENCE_THRESHOLD
        ):
            try:
                # Weather is requested only for sightings
                # that will be saved.
                weather_data = get_current_weather(
                    latitude=latitude,
                    longitude=longitude
                )

                (
                    sighting_id,
                    image_filename,
                    saved_image_path
                ) = save_sighting(
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

                print(
                    "SIGHTING SAVED:",
                    sighting_id
                )

                print(
                    "IMAGE PATH:",
                    saved_image_path
                )

                print(
                    "CSV FILE:",
                    CSV_FILE
                )

            except Exception as error:
                print(
                    "SAVING ERROR:",
                    str(error)
                )

                traceback.print_exc()

                return jsonify({
                    "error": (
                        "Prediction succeeded, but weather "
                        "retrieval or saving failed: "
                        f"{str(error)}"
                    ),
                    "label": label,
                    "confidence": confidence,
                    "saved": False
                }), 500

        else:
            print(
                "Image was not saved because it was not "
                "a confident Monarch prediction."
            )

        # -------------------------------------------------
        # 8. Send JSON response to App Inventor
        # -------------------------------------------------

        response = {
            "label": label,
            "confidence": confidence,
            "raw_score": raw_prediction,
            "saved": saved,
            "sighting_id": sighting_id,
            "image_filename": image_filename,
            "latitude": (
                latitude
                if location_valid
                else None
            ),
            "longitude": (
                longitude
                if location_valid
                else None
            ),
            "location_valid": location_valid,
            "timestamp": timestamp,
            "weather": weather_data
        }

        print("RETURNING SUCCESS RESPONSE")
        print("========================================")

        return jsonify(response), 200

    except RequestEntityTooLarge:
        print("REQUEST ENTITY TOO LARGE - see traceback for exact cause:")
        traceback.print_exc()
        # Let this propagate to the @app.errorhandler(413) below
        # instead of being reported as a generic 500 error.
        raise

    except Exception as error:
        print(
            "UNEXPECTED SERVER ERROR:",
            str(error)
        )

        traceback.print_exc()

        return jsonify({
            "error": (
                f"Unexpected server error: {str(error)}"
            ),
            "saved": False
        }), 500


# =========================================================
# ERROR HANDLERS
# =========================================================

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
        "instructions": (
            "Send a POST request to /predict."
        )
    }), 405

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "The uploaded image is too large. Maximum size is 50 MB.",
        "saved": False
    }), 413
# =========================================================
# START SERVER
# =========================================================


if __name__ == "__main__":
    app.run(
        debug=False,
        use_reloader=False,
        host="0.0.0.0",
        port=5000
    )
