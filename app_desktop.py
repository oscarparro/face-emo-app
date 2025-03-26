# app_desktop.py

import sys
import os
import pickle
import cv2
import face_recognition
import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QSizePolicy
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

EMBEDDINGS_FILE = "embeddings.pkl"

# Cargamos (o creamos) las listas de rostros conocidos.
if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
else:
    known_face_encodings = []
    known_face_names = []


def save_embeddings():
    """Guarda en disco los rostros conocidos."""
    data = {
        "encodings": known_face_encodings,
        "names": known_face_names
    }
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(data, f)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Live Face Recognition - Desktop App")
        self.setGeometry(0, 0, 1920, 1080)

# ------------------------------------------------
        # 1) Widget central y layout principal
        # ------------------------------------------------
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # ------------------------------------------------
        # 2) Título "grande" en la parte superior
        # ------------------------------------------------
        self.title_label = QLabel("MI APLICACIÓN DE RECONOCIMIENTO FACIAL")
        self.title_label.setAlignment(Qt.AlignCenter)
        # Ajustes de fuente con StyleSheet
        self.title_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; margin: 10px;"
        )
        self.layout.addWidget(self.title_label)

        # ------------------------------------------------
        # 3) Label para el video
        # ------------------------------------------------
        self.video_label = QLabel("Aquí se mostrará la cámara")
        self.video_label.setAlignment(Qt.AlignCenter)
        # Aumenta el tamaño mínimo (ej. 640x480)
        self.video_label.setMinimumSize(640, 480)
        # Opcional: permitir que el video_label crezca
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.video_label)

        # ------------------------------------------------
        # 4) Layout para los botones (horizontal)
        # ------------------------------------------------
        btn_layout = QHBoxLayout()

        self.btn_register = QPushButton("Registrar Rostro")
        # Limita su ancho máximo para que no se estiren
        self.btn_register.setMaximumWidth(150)
        # También puedes añadir un estilo de botón
        self.btn_register.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "border-radius: 5px; padding: 8px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        self.btn_register.clicked.connect(self.register_face)
        btn_layout.addWidget(self.btn_register)

        self.btn_quit = QPushButton("Salir")
        self.btn_quit.setMaximumWidth(150)
        self.btn_quit.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "border-radius: 5px; padding: 8px; }"
            "QPushButton:hover { background-color: #da190b; }"
        )
        self.btn_quit.clicked.connect(self.close_app)
        btn_layout.addWidget(self.btn_quit)

        # Si quieres que los botones estén alineados a la derecha,
        # añade un stretch a la izquierda:
        # btn_layout.insertStretch(0, 1)

        self.layout.addLayout(btn_layout)

        # ------------------------------------------------
        # 5) Inicializar cámara y QTimer para refrescar
        # ------------------------------------------------
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo acceder a la cámara.")
            sys.exit(-1)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(60)  # ~60 fps

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model = 'hog')
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Para cada cara detectada
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Desconocido"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches and matches[best_match_index]:
                        name = known_face_names[best_match_index]

                # Dibujar bounding box y etiqueta
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error en detección: {e}")                   

        # Convertir de nuevo a formato QImage para mostrarlo
        # frame sigue en BGR, convertimos a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Escalar con la relación de aspecto
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(
            self.video_label.size(),           # Tamaño disponible en el label
            Qt.KeepAspectRatio,                # Mantiene la relación de aspecto
            Qt.SmoothTransformation            # Escalado de buena calidad
        )

        # Mostrar en el label
        self.video_label.setPixmap(pix)

    def register_face(self):
        """
        Al pulsar el botón 'Registrar Rostro':
         1) Pedir el nombre
         2) Tomar el frame actual de la cámara
         3) Si se detecta UNA cara, extraer su encoding
         4) Guardar en la base de datos (pickle)
        """
        name, ok = QInputDialog.getText(self, "Registrar Rostro", "Introduce tu nombre:")
        if not ok or not name.strip():
            return  # usuario canceló o no escribió nada

        # Tomar un frame de la cámara AHORA
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.warning(self, "Aviso", "No se pudo capturar frame de la cámara.")
            return

        # Convertir a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar y codificar
        try:
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Se produjo un error en face_recognition: {e}")
            return

        if len(face_encodings) == 1:
            # Tenemos exactamente una cara
            new_encoding = face_encodings[0]
            known_face_encodings.append(new_encoding)
            known_face_names.append(name.strip())
            save_embeddings()
            QMessageBox.information(self, "OK", f"Rostro registrado con el nombre: {name}")
        elif len(face_encodings) == 0:
            QMessageBox.warning(self, "Aviso", "No se detectó ninguna cara. Acércate o revisa iluminación.")
        else:
            QMessageBox.warning(self, "Aviso", "Se detectaron varias caras. Asegúrate de estar solo en la imagen.")

    def close_app(self):
        self.cap.release()
        self.close()

    def closeEvent(self, event):
        """Se llama cuando se cierra la ventana."""
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()
