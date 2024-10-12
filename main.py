import cv2


class EyeDetectorHaarCascade:
    def __init__(self):
        # Carregar o classificador Haar Cascade para rosto e olhos
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def detect_eyes(self, frame):
        # Converter a imagem para escala de cinza
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostos
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # Para cada rosto detectado, tentar detectar olhos
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Desenhar um retângulo ao redor do rosto
            roi_gray = gray_frame[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Detectar olhos dentro da região do rosto detectado
            eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=15)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0),
                              2)  # Desenhar um retângulo ao redor dos olhos

    def run(self):
        # Inicializar a captura da webcam
        camera = cv2.VideoCapture(0)

        while True:
            ret, frame = camera.read()
            if not ret:
                break

            # Espelhar o frame horizontalmente
            frame = cv2.flip(frame, 1)

            # Detectar olhos no frame
            self.detect_eyes(frame)

            # Exibir o frame com as detecções
            cv2.imshow("Eye Detection (Haar Cascade)", frame)

            # Sair do loop se a tecla 'q' for pressionada
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Liberar a câmera e fechar as janelas
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    eye_detector = EyeDetectorHaarCascade()
    eye_detector.run()
