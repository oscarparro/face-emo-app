import sys
import os
import pickle
import cv2
import face_recognition
import numpy as np
import hashlib
import datetime

from mtcnn import MTCNN

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QInputDialog, QMessageBox, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import QTimer, Qt, QElapsedTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QColor, QIcon

EMBEDDINGS_FILE = "embeddings.pkl"

def load_registrations():
    """Carga la lista de registros (lista de diccionarios) desde embeddings.pkl."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    else:
        return []

def save_registrations(registrations):
    """Guarda la lista de registros en embeddings.pkl."""
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(registrations, f)

def generate_color(name):
    if name == "Desconocido":
        return (0, 255, 0)
    h = hashlib.md5(name.encode()).hexdigest()
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

class EmbeddingWorker(QThread):
    """
    Worker para calcular embeddings usando MTCNN.
    Recibe un frame y la lista actual de registros, y emite una lista de tuplas:
    (top, right, bottom, left, nombre)
    """
    result_ready = Signal(list)

    def __init__(self, frame, registrations, parent=None):
        super().__init__(parent)
        self.frame = frame.copy()
        self.registrations = registrations

    def run(self):
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        detector = MTCNN()
        detections = detector.detect_faces(rgb_frame)
        results = []
        height, width, _ = rgb_frame.shape

        # Extraer embeddings y nombres de los registros actuales
        registered_encodings = [reg["embedding"] for reg in self.registrations]
        registered_names = [reg["name"] for reg in self.registrations]

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
            # Verificar tamaño mínimo
            if face_image.shape[0] < 20 or face_image.shape[1] < 20:
                continue

            try:
                face_encodings = face_recognition.face_encodings(
                    face_image,
                    known_face_locations=[(0, face_image.shape[1], face_image.shape[0], 0)]
                )
            except TypeError:
                face_encodings = face_recognition.face_encodings(face_image)

            name = "Desconocido"
            if face_encodings:
                face_encoding = face_encodings[0]
                if registered_encodings:
                    matches = face_recognition.compare_faces(registered_encodings, face_encoding, tolerance=0.6)
                    if any(matches):
                        face_distances = face_recognition.face_distance(registered_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = registered_names[best_match_index]
            results.append((top, right, bottom, left, name))
        self.result_ready.emit(results)

class InfoWindow(QMainWindow):
    """
    Ventana para mostrar la información de registros en una tabla.
    Columnas: Imagen, Nombre, Color, Fecha, Acciones (botón Eliminar).
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Información de Registros")
        self.resize(800, 400)
        self.main_window = main_window
        self.setup_ui()

    def setup_ui(self):
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Imagen", "Nombre", "Color", "Fecha", "Acciones"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setCentralWidget(self.table)
        self.populate_table()

    def populate_table(self):
        registrations = self.main_window.registrations_data
        self.table.setRowCount(len(registrations))
        for row, info in enumerate(registrations):
            # Columna Imagen
            image_item = QTableWidgetItem()
            if os.path.exists(info['image_path']):
                pixmap = QPixmap(info['image_path']).scaled(50, 50, Qt.KeepAspectRatio)
                image_item.setIcon(QIcon(pixmap))
            image_item.setText(os.path.basename(info['image_path']))
            self.table.setItem(row, 0, image_item)

            # Columna Nombre
            name_item = QTableWidgetItem(info['name'])
            self.table.setItem(row, 1, name_item)

            # Columna Color
            color_item = QTableWidgetItem(f"RGB{info['color']}")
            r, g, b = info['color']
            color_item.setBackground(QColor(r, g, b))
            self.table.setItem(row, 2, color_item)

            # Columna Fecha
            date_item = QTableWidgetItem(info['date'])
            self.table.setItem(row, 3, date_item)

            # Columna Acciones: Botón Eliminar
            btn_delete = QPushButton("Eliminar")
            btn_delete.setStyleSheet("background-color: #f44336; color: white;")
            btn_delete.clicked.connect(lambda checked, row=row: self.delete_row(row))
            self.table.setCellWidget(row, 4, btn_delete)

    def delete_row(self, index):
        self.main_window.delete_registration(index)
        self.populate_table()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Face Recognition - Desktop App")
        self.setGeometry(0, 0, 1280, 720)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.title_label = QLabel("MI APLICACIÓN DE RECONOCIMIENTO FACIAL")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        self.main_layout.addWidget(self.title_label)

        self.content_layout = QHBoxLayout()
        self.main_layout.addLayout(self.content_layout)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.content_layout.addWidget(self.video_label)

        self.legend_label = QLabel("<b>Personas detectadas:</b><br>No hay nadie")
        self.legend_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.legend_label.setFixedWidth(300)
        self.legend_label.setStyleSheet("font-size: 14px; padding: 10px;")
        self.content_layout.addWidget(self.legend_label)

        btn_layout = QHBoxLayout()

        self.btn_register = QPushButton("Registrar Rostro")
        self.btn_register.setMaximumWidth(150)
        self.btn_register.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; border-radius: 5px; padding: 8px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        self.btn_register.clicked.connect(self.register_face)
        btn_layout.addWidget(self.btn_register)

        self.btn_info = QPushButton("Información")
        self.btn_info.setMaximumWidth(150)
        self.btn_info.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; border-radius: 5px; padding: 8px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        self.btn_info.clicked.connect(self.open_info)
        btn_layout.addWidget(self.btn_info)

        self.btn_quit = QPushButton("Salir")
        self.btn_quit.setMaximumWidth(150)
        self.btn_quit.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; border-radius: 5px; padding: 8px; }"
            "QPushButton:hover { background-color: #da190b; }"
        )
        self.btn_quit.clicked.connect(self.close_app)
        btn_layout.addWidget(self.btn_quit)

        self.main_layout.addLayout(btn_layout)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo acceder a la cámara.")
            sys.exit(-1)

        self.last_process_time = QElapsedTimer()
        self.last_process_time.start()

        # Cargar registros (lista de diccionarios)
        self.registrations_data = load_registrations()

        self.color_map = {"Desconocido": (0, 255, 0)}
        self.recent_faces = []  # (top, right, bottom, left, nombre)

        self.embedding_worker = None
        self.mtcnn_detector = MTCNN()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(60)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_frame.shape

        detections = self.mtcnn_detector.detect_faces(rgb_frame)
        face_locations = []
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
            face_locations.append((top, right, bottom, left))

        for (top, right, bottom, left) in face_locations:
            name = "Desconocido"
            for (t, r, b, l, n) in self.recent_faces:
                if abs(t - top) < 30 and abs(r - right) < 30 and abs(b - bottom) < 30 and abs(l - left) < 30:
                    name = n
                    break
            color = self.color_map.get(name, (0, 255, 0))
            width_box = right - left
            height_box = bottom - top
            pad_w = int(width_box * 0.05)
            pad_h = int(height_box * 0.05)
            pt1 = (max(0, left - pad_w), max(0, top - pad_h))
            pt2 = (min(frame.shape[1], right + pad_w), min(frame.shape[0], bottom + pad_h))
            cv2.rectangle(frame, pt1, pt2, color, 2)

        if self.last_process_time.elapsed() > 10000:
            if self.embedding_worker is None or not self.embedding_worker.isRunning():
                self.embedding_worker = EmbeddingWorker(frame, self.registrations_data)
                self.embedding_worker.result_ready.connect(self.update_faces_worker)
                self.embedding_worker.start()
                self.last_process_time.restart()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_frame, w_frame, ch = frame_rgb.shape
        bytes_per_line = ch * w_frame
        qimg = QImage(frame_rgb.data, w_frame, h_frame, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def update_faces_worker(self, results):
        self.recent_faces = results
        self.update_legend()

    def update_legend(self):
        if not self.recent_faces:
            self.legend_label.setText("<b>Personas detectadas:</b><br>No hay nadie")
            return
        html = "<b>Personas detectadas:</b><br>"
        names_seen = set()
        for _, _, _, _, name in self.recent_faces:
            if name not in names_seen:
                names_seen.add(name)
                color = self.color_map.get(name, (0, 255, 0))
                color_hex = '#%02x%02x%02x' % color
                html += f'<span style="color:{color_hex};">&#9632;</span> {name}<br>'
        self.legend_label.setText(html)

    def register_face(self):
        name, ok = QInputDialog.getText(self, "Registrar Rostro", "Introduce tu nombre:")
        if not ok or not name.strip():
            return

        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.warning(self, "Aviso", "No se pudo capturar frame de la cámara.")
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_frame.shape
        try:
            detector = MTCNN()
            detections = detector.detect_faces(rgb_frame)
            face_encodings = []
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
                    face_encodings_found = face_recognition.face_encodings(
                        face_image,
                        known_face_locations=[(0, face_image.shape[1], face_image.shape[0], 0)]
                    )
                except TypeError:
                    face_encodings_found = face_recognition.face_encodings(face_image)
                if face_encodings_found:
                    face_encodings.append(face_encodings_found[0])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Se produjo un error en face_recognition: {e}")
            return

        if len(face_encodings) == 1:
            new_encoding = face_encodings[0]
            os.makedirs("registered_faces", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"registered_faces/{name.strip()}_{timestamp}.png"
            cv2.imwrite(image_filename, frame)

            color = generate_color(name.strip())
            date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            new_registration = {
                "embedding": new_encoding,
                "name": name.strip(),
                "image_path": image_filename,
                "color": color,
                "date": date_str
            }
            self.registrations_data.append(new_registration)
            save_registrations(self.registrations_data)

            if name.strip() not in self.color_map:
                self.color_map[name.strip()] = color

            QMessageBox.information(self, "OK", f"Rostro registrado con el nombre: {name}")
        elif len(face_encodings) == 0:
            QMessageBox.warning(self, "Aviso", "No se detectó ninguna cara. Acércate o revisa iluminación.")
        else:
            QMessageBox.warning(self, "Aviso", "Se detectaron varias caras. Asegúrate de estar solo en la imagen.")

    def delete_registration(self, index):
        try:
            del self.registrations_data[index]
            save_registrations(self.registrations_data)
            QMessageBox.information(self, "Información", "Registro eliminado correctamente.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"No se pudo eliminar el registro: {e}")

    def open_info(self):
        if not self.registrations_data:
            QMessageBox.information(self, "Información", "No hay registros disponibles.")
            return
        self.info_window = InfoWindow(self)
        self.info_window.show()

    def close_app(self):
        self.cap.release()
        self.close()

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

def main():
    # Asegúrate de eliminar o renombrar el embeddings.pkl antiguo para usar el nuevo formato.
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
