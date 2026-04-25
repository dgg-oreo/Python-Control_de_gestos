import cv2
import mediapipe as mp
import serial
import time
import urllib.request
import os
import numpy as np
from enum import Enum

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )

PUERTO = '/dev/ttyACM0'
BAUD_RATE =9600 //velocidad de modulacion
  
try:
    arduino = serial.Serial(PUERTO, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Arduino conectado en {PUERTO}")
except serial.SerialException:
    print(f"No se pudo conectar al Arduino en {PUERTO}")
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

class Modo(Enum):
    LEDS  = "LEDs"
    SERVO = "Servo"
    MOTOR = "Motor DC"

MODOS    = list(Modo)
modo_idx = 0

leds_estado  = [False, False, False, False]
servo_estado = "Cerrado"
motor_estado = "Detenido"

ultimo_cmd = 0
DELAY_CMD  = 1.2

ultimo_led = [0, 0, 0, 0]
DELAY_LED  = 0.3

PUNTAS = [8, 12, 16, 20]

def contar_dedos(lm):
    dedos = [1 if lm[4].x < lm[3].x else 0]
    for p in PUNTAS:
        dedos.append(1 if lm[p].y < lm[p - 2].y else 0)
    return sum(dedos), dedos

def enviar(cmd: str):
    global ultimo_cmd
    now = time.time()
    if now - ultimo_cmd < DELAY_CMD:
        return False
    arduino.write((cmd + "\n").encode())
    ultimo_cmd = now
    print(f"  CMD: {cmd}")
    return True

def procesar_leds(dedos_lista):
    global leds_estado, ultimo_led
    now = time.time()
    for i, dedo_idx in enumerate([1, 2, 3, 4]):
        nuevo = bool(dedos_lista[dedo_idx])
        if nuevo != leds_estado[i] and (now - ultimo_led[i]) >= DELAY_LED:
            leds_estado[i] = nuevo
            ultimo_led[i]  = now
            cmd = f"LED{i+1}_ON" if nuevo else f"LED{i+1}_OFF"
            arduino.write((cmd + "\n").encode())
            print(f"  CMD: {cmd}")

def procesar_servo(n_dedos):
    global servo_estado
    if n_dedos == 5 and servo_estado != "Abierto":
        if enviar("Abrir"):
            servo_estado = "Abierto"
    elif n_dedos == 0 and servo_estado != "Cerrado":
        if enviar("Cerrar"):
            servo_estado = "Cerrado"

def procesar_motor(n_dedos):
    global motor_estado
    if n_dedos >= 3 and motor_estado != "Adelante":
        if enviar("MotorAdelante"):
            motor_estado = "Adelante"
    elif n_dedos == 1 and motor_estado != "Cerrando":
        if enviar("MotorCerrar"):
            motor_estado = "Cerrando"
    elif n_dedos == 0 and motor_estado != "Detenido":
        if enviar("MotorDetener"):
            motor_estado = "Detenido"


BG = (10,  10,  10)   # negro casi puro
PANEL = (18,  18,  18)   # panel ligeramente más claro
WHITE = (240, 240, 240)  # blanco principal
GRAY = (90,  90,  90)   # gris medio
DGRAY = (35,  35,  35)   # gris oscuro (bordes/fondos)
LGRAY = (160, 160, 160)  # gris claro

PANEL_W = 280
CAM_W = 640
CAM_H = 480
WIN_W = CAM_W + PANEL_W
WIN_H = CAM_H

def tx(img, txt, x, y, sc=0.45, col=WHITE, th=1):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)

def hrule(img, x, y, w, col=DGRAY):
    cv2.line(img, (x, y), (x + w, y), col, 1)

def pill(img, x, y, w, h, col, filled=False):
    """Rectángulo de bordes redondeados simples."""
    r = 5
    if filled:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), col, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), col, -1)
    else:
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 1)


CONEXIONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

def dibujar(frame, n_dedos, dedos_lista, modo_actual):
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    canvas[:] = BG

    canvas[0:CAM_H, 0:CAM_W] = cv2.resize(frame, (CAM_W, CAM_H))
    cv2.rectangle(canvas, (0, 0), (CAM_W - 1, CAM_H - 1), DGRAY, 1)

   
    cv2.rectangle(canvas, (12, 12), (200, 38), BG, -1)
    cv2.rectangle(canvas, (12, 12), (200, 38), GRAY, 1)
    tx(canvas, f"MODO: {modo_actual.value.upper()}", 20, 31, 0.45, WHITE, 1)


    cv2.rectangle(canvas, (12, 46), (130, 68), BG, -1)
    tx(canvas, f"DEDOS: {n_dedos}", 20, 63, 0.45, LGRAY)

    px = CAM_W
    cv2.rectangle(canvas, (px, 0), (WIN_W, WIN_H), PANEL, -1)
    cv2.line(canvas, (px, 0), (px, WIN_H), DGRAY, 1)

    # Cabecera
    tx(canvas, "CONTROL", px + 20, 32, 0.65, WHITE, 1)
    tx(canvas, "por gestos", px + 20, 50, 0.38, GRAY)
    hrule(canvas, px + 10, 62, PANEL_W - 20, DGRAY)

    # Modo activo
    y = 80
    tx(canvas, "MODO", px + 20, y, 0.35, GRAY)
    y += 18
    pill(canvas, px + 14, y, PANEL_W - 28, 30, DGRAY, filled=True)
    pill(canvas, px + 14, y, PANEL_W - 28, 30, GRAY)
    tx(canvas, modo_actual.value.upper(), px + 22, y + 20, 0.55, WHITE, 1)
    y += 40
    tx(canvas, "[ A ] prev    [ D ] sig", px + 20, y, 0.33, GRAY)

    hrule(canvas, px + 10, y + 14, PANEL_W - 20)
    y += 28


    if modo_actual == Modo.LEDS:
        nombres = ["LED 1", "LED 2", "LED 3", "LED 4"]
        dedos_n = ["Indice", "Medio", "Anular", "Menique"]
        for i, nombre in enumerate(nombres):
            enc = leds_estado[i]
            col_ind = WHITE if enc else DGRAY
            col_txt = WHITE if enc else GRAY
            # Indicador
            cv2.circle(canvas, (px + 26, y + 10), 6,
                       WHITE if enc else DGRAY,
                       -1 if enc else 1)
            tx(canvas, nombre, px + 42, y + 15, 0.44, col_txt)
            tx(canvas, dedos_n[i], px + 120, y + 15, 0.35, GRAY)
            state_txt = "ON " if enc else "OFF"
            tx(canvas, state_txt, px + PANEL_W - 46, y + 15, 0.42,
               WHITE if enc else GRAY, 1 if enc else 1)
            y += 30

        hrule(canvas, px + 10, y + 4, PANEL_W - 20)
        y += 16
        tx(canvas, "Levanta cada dedo para", px + 20, y, 0.35, GRAY)
        y += 16
        tx(canvas, "encender su LED.", px + 20, y, 0.35, GRAY)

    elif modo_actual == Modo.SERVO:
        abierto  = servo_estado == "Abierto"
        col_s    = WHITE if abierto else GRAY
        pill(canvas, px + 14, y, PANEL_W - 28, 38, DGRAY, filled=True)
        if abierto:
            pill(canvas, px + 14, y, PANEL_W - 28, 38, WHITE)
        tx(canvas, servo_estado.upper(), px + 22, y + 25, 0.7, col_s, 2)
        y += 52

        gestos = [("5 dedos", "ABRIR"), ("0 dedos", "CERRAR")]
        for g, a in gestos:
            tx(canvas, g, px + 22, y, 0.4, GRAY)
            tx(canvas, a, px + 130, y, 0.4, WHITE)
            y += 22

    elif modo_actual == Modo.MOTOR:
        col_m = WHITE if motor_estado != "Detenido" else GRAY
        pill(canvas, px + 14, y, PANEL_W - 28, 38, DGRAY, filled=True)
        if motor_estado != "Detenido":
            pill(canvas, px + 14, y, PANEL_W - 28, 38, WHITE)
        tx(canvas, motor_estado.upper(), px + 22, y + 25, 0.6, col_m, 2)
        y += 52

        gestos_m = [("3-5 dedos", "ADELANTE"), ("1 dedo", "REVERSA"), ("0 dedos", "DETENER")]
        for g, a in gestos_m:
            tx(canvas, g, px + 22, y, 0.4, GRAY)
            tx(canvas, a, px + 130, y, 0.4, WHITE)
            y += 22


    return canvas


print("Control por Gestos  |  A/D = modo  |  Q = salir")

cv2.namedWindow("Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Control", WIN_W, WIN_H)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    modo_actual = MODOS[modo_idx]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    n_dedos = 0
    dedos_list = [0] * 5

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        n_dedos, dedos_list = contar_dedos(lm)

        h, w, _ = frame.shape
        pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(21)]
        for a, b in CONEXIONES:
            cv2.line(frame, pts[a], pts[b], LGRAY, 1)
        for i, (px_l, py_l) in enumerate(pts):
            col = WHITE if i in [4, 8, 12, 16, 20] else GRAY
            cv2.circle(frame, (px_l, py_l), 4, col, -1)

        if modo_actual == Modo.LEDS:
            procesar_leds(dedos_list)
        elif modo_actual == Modo.SERVO:
            procesar_servo(n_dedos)
        elif modo_actual == Modo.MOTOR:
            procesar_motor(n_dedos)

    canvas = dibujar(frame, n_dedos, dedos_list, modo_actual)
    cv2.imshow("Control", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        modo_idx = (modo_idx + 1) % len(MODOS)
    elif key == ord('a'):
        modo_idx = (modo_idx - 1) % len(MODOS)

cap.release()
cv2.destroyAllWindows()
arduino.close()
print("Cerrado.")