# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
from mtcnn import MTCNN
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

def decode_image(image_b64):
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

@app.route('/process_embedding', methods=['POST'])
def process_embedding():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No se proporcionó la imagen"}), 400

    image_b64 = data["image"]
    frame = decode_image(image_b64)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    detections = detector.detect_faces(rgb_frame)
    results = []
    height, width, _ = rgb_frame.shape

    for detection in detections:
        if detection['confidence'] < 0.90:
            continue
        x, y, w, h = detection['box']
        top = max(0, y)
        left = max(0, x)
        right = min(width, x + w)
        bottom = min(height, y + h)
        if bottom <= top or right <= left:
            continue

        face_image = rgb_frame[top:bottom, left:right]
        if face_image.shape[0] < 20 or face_image.shape[1] < 20:
            continue

        try:
            face_encodings = face_recognition.face_encodings(
                face_image,
                known_face_locations=[(0, face_image.shape[1], face_image.shape[0], 0)]
            )
        except TypeError:
            face_encodings = face_recognition.face_encodings(face_image)
        if face_encodings:
            results.append({
                "box": [top, right, bottom, left],
                "embedding": face_encodings[0].tolist()
            })
    return jsonify({"detections": results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
