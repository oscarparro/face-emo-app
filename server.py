# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
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

    # Usamos el modelo 'hog' (sin MTCNN)
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    detections = []
    for loc, encoding in zip(face_locations, face_encodings):
        # loc es (top, right, bottom, left)
        detections.append({
            "box": list(loc),
            "embedding": encoding.tolist()
        })
    return jsonify({"detections": detections})

if __name__ == '__main__':
    # Escucha en todas las interfaces en el puerto 5001
    app.run(host='0.0.0.0', port=5001)
