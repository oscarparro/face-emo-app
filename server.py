# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
import base64
import io
from PIL import Image
import os
import pickle

app = Flask(__name__)
CORS(app)

# Archivo de registros en el servidor (se asume que ya se han registrado rostros previamente)
EMBEDDINGS_FILE = "embeddings.pkl"

def load_registrations():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
        # Se espera un diccionario con claves "encodings" y "names"
        return data
    else:
        return {"encodings": [], "names": []}

# Cargar registros (esto supone que el servidor tiene acceso a la base de datos de registros)
registrations = load_registrations()

def decode_image(image_b64):
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

@app.route('/process_embedding', methods=['POST'])
def process_embedding():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No se proporcionó la imagen"}), 400

    # La imagen recibida ya es el recorte del bounding box
    face_crop = decode_image(data["image"])
    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    
    # Detectar la cara en el recorte (idealmente, solo hay una)
    face_locations = face_recognition.face_locations(rgb_face, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_face, face_locations)
    
    if len(face_encodings) == 0:
        return jsonify({"name": "Desconocido"})
    embedding = face_encodings[0]
    
    # Comparar con registros existentes
    if registrations["encodings"]:
        matches = face_recognition.compare_faces(registrations["encodings"], embedding, tolerance=0.6)
        if any(matches):
            distances = face_recognition.face_distance(registrations["encodings"], embedding)
            best_match_index = np.argmin(distances)
            if matches[best_match_index]:
                recognized_name = registrations["names"][best_match_index]
                return jsonify({"name": recognized_name})
    return jsonify({"name": "Desconocido"})

if __name__ == '__main__':
    # Escucha en todas las interfaces en el puerto 5001
    app.run(host='0.0.0.0', port=5001)
