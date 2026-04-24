import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import serial
import time
import urllib.request
import os

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("⬇️  Descargando modelo de detección de manos...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("✅ Modelo descargado")

PUERTO = '/dev/ttyACM0'
BAUD_RATE = 9600

try:
    arduino = serial.Serial(PUERTO, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Si se conecto el arduino en {PUERTO}")
except serial.SerialException:
    print(f"No se ha podido conectar al arduino en {PUERTO}")
    exit()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7
)

detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
estado_actual = None

def contar_dedos(landmarks):
    dedos = []
    puntas = [8, 12, 16, 20]

    # Pulgar
    if landmarks[4].x < landmarks[3].x:
        dedos.append(1)
    else:
        dedos.append(0)

    # Los otros dedos
    for punta in puntas:
        if landmarks[punta].y < landmarks[punta - 2].y:
            dedos.append(1)
        else:
            dedos.append(0)

    return sum(dedos)

print("🖐 Muestra tu mano:")
print("   ✋ Mano abierta (5 dedos) = ABRIR servo")
print("   ✊ Puño cerrado (0 dedos) = CERRAR servo")
print("   Presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    gesto = "Sin mano"

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        dedos = contar_dedos(landmarks)

        h, w, _ = frame.shape
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        if dedos == 5 and estado_actual != "ABRIR":
            arduino.write(b"ABRIR\n")
            estado_actual = "ABRIR"
            gesto = "ABRIENDO"
            print("→ Comando enviado: ABRIR")
        elif dedos == 0 and estado_actual != "CERRAR":
            arduino.write(b"CERRAR\n")
            estado_actual = "CERRAR"
            gesto = "CERRANDO"
            print("→ Comando enviado: CERRAR")
        else:
            gesto = f"Dedos: {dedos}"

    cv2.putText(frame, gesto, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, f"Estado: {estado_actual}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("Control por Gestos - Servo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
print("👋 Programa cerrado")