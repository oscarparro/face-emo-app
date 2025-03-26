# face_utils.py

import os
import pickle
import face_recognition
import numpy as np
import cv2

EMBEDDINGS_FILE = "embeddings.pkl"

def load_embeddings():
    """
    Carga del archivo EMBEDDINGS_FILE las listas de nombres y encodings si existen.
    Retorna (known_face_encodings, known_face_names).
    """
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
        return data["encodings"], data["names"]
    else:
        return [], []


def save_embeddings(known_face_encodings, known_face_names):
    """
    Guarda los encodings y nombres en EMBEDDINGS_FILE (pickle).
    """
    data = {
        "encodings": known_face_encodings,
        "names": known_face_names
    }
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(data, f)


def detect_and_recognize_faces(frame, known_face_encodings, known_face_names, tolerance=0.6):
    """
    - Dado un frame BGR de OpenCV, lo convierte a RGB, 
      detecta caras y calcula sus encodings.
    - Compara con la base de datos de rostros conocidos (known_face_encodings, known_face_names).
    - Retorna una lista de:
      [ (top, right, bottom, left, nombre) , ... ]

    :param frame: imagen en formato BGR (numpy array)
    :param known_face_encodings: lista de encodings de caras conocidas
    :param known_face_names: lista de nombres correspondientes
    :param tolerance: umbral para face_recognition.compare_faces
    :return: lista de tuplas (top, right, bottom, left, name_detected)
    """
    # Convertir BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar rostros
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compara con caras conocidas
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
        name = "Desconocido"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Guardamos la info en la lista
        face_results.append((top, right, bottom, left, name))

    return face_results


def register_face(frame, name, known_face_encodings, known_face_names):
    """
    Registra una nueva cara:
      - Toma el frame BGR,
      - Convierte a RGB,
      - Detecta si hay UNA cara,
      - Obtiene su encoding,
      - Lo añade a known_face_encodings y known_face_names,
      - Llama a save_embeddings() para persistir.

    Retorna True/False dependiendo de si la operación fue exitosa.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if len(face_encodings) == 1:
        new_encoding = face_encodings[0]
        known_face_encodings.append(new_encoding)
        known_face_names.append(name)
        save_embeddings(known_face_encodings, known_face_names)
        return True
    else:
        return False
