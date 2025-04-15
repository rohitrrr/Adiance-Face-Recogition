# Adiance Face Recognition System

A face recognition system using RetinaFace for face detection and AdaFace for face recognition.

## Features

- Face detection using RetinaFace
- Face recognition using AdaFace
- Web interface for enrollment and identification
- REST API for programmatic access
- Rate limiting and security features

## Requirements

- Python 3.7 or higher
- OpenCV
- ONNX Runtime
- Flask
- NumPy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd flask_app
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Configuration

The system uses two main models:
1. RetinaFace (`retinaface_mnet_v2.onnx`) for face detection
2. AdaFace (`Embedding.onnx`) for face recognition

Make sure these model files are in the root directory of the project.

## Model Files
Due to size limitations, the following model files are not included in this repository:
- `Embedding.onnx` (249.51 MB)
- `retinaface_mnet_v2.onnx`

Please contact the repository maintainer to obtain these files and place them in the root directory of the project.

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## API Endpoints

### Enrollment
- POST `/api/enroll`
  - Request: Form data with `file` (image) and `subject_id`
  - Response: JSON with enrollment status

### Identification
- POST `/api/search`
  - Request: Form data with `file` (image)
  - Response: JSON with matching candidates and scores

## Directory Structure

```
flask_app/
├── app.py              # Main Flask application
├── adiance_wrapper.py  # Python wrapper for face recognition
├── config/             # Configuration files
├── static/             # Static files (CSS, JS, uploads)
├── templates/          # HTML templates
└── enrollment_db/      # Enrollment database
```

## License

MIT License 