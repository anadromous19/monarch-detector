# imonarch-detector 
Monarch identification app using MIT App Inventor and Flask

Monarch Detector is a mobile application that uses a convolutional
neural network to classify a photographed butterfly as a monarch or
non-monarch. The project combines an MIT App Inventor mobile client
with a Flask and TensorFlow inference API hosted on AWS.

## Project Goal

The goal of this project is to make monarch identification and
conservation reporting more accessible. Users can photograph a
butterfly, receive a prediction with a confidence score, and contribute
environmental information associated with qualifying sightings.

The application is an educational conservation tool and should not be
treated as a replacement for verification by an experienced observer.

## Current Features

- Captures butterfly photographs using the device camera
- Sends photographs securely using HTTPS
- Classifies images as Monarch or Non-Monarch
- Displays a confidence score
- Collects approximate location information
- Retrieves weather information for qualifying sightings
- Stores sighting images in Amazon S3
- Stores sighting records in Amazon DynamoDB

## Model Performance

The convolutional neural network was evaluated on a held-out test set.

| Metric | Result |
| --- | ---: |
| Test accuracy | 83.5% |
| Monarch recall | 82.5% |
| F1 score | Approximately 0.82 |
| Training accuracy | Approximately 88% |

Performance may vary with image quality, lighting, camera angle,
background, and butterfly species.

## Technology

### Mobile application

- MIT App Inventor
- Camera component
- Location Sensor
- Web component

### Backend

- Python
- Flask
- TensorFlow and Keras
- Pillow
- Gunicorn
- Nginx

### Cloud services

- Amazon EC2
- Amazon S3
- Amazon DynamoDB
- HTTPS certificate managed with Certbot

## How It Works

1. The user photographs a butterfly.
2. The mobile app sends the image and approximate coordinates to the API.
3. The API validates and preprocesses the image.
4. The TensorFlow model generates a prediction.
5. The API returns the label and confidence score.
6. Qualifying sightings are enriched with weather information and stored.

See [architecture.md](architecture.md) for additional technical details.

## Repository Structure

```text
.
├── README.md
├── architecture.md
├── app_inventor/
│   └── Monarch_detector.aia
├── flask_backend/
│   ├── app.py
│   └── requirements.txt
├── docs/
│   ├── habitat-data-fields.md
│   └── images/
└── .gitignore