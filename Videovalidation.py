import socket
import time
import cv2
from ultralytics import YOLO
import depthai as dai
import numpy as np

# Configuración del servidor
host = 'localhost'
port = 12345

# Iniciar el servidor
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)

print('Servidor Python esperando conexiones en el puerto', port)
conn, addr = server_socket.accept()
print('Conectado a:', addr)

# Parámetros de la cámara
anguloFovMed = 18.3
anguloFovMed2 = 30.11

# Cargar el modelo YOLOv8
model = YOLO('runs/detect/train/weights/best.pt')

# Crear pipeline para la cámara y la profundidad
pipeline = dai.Pipeline()

# Crear nodos de cámara y profundidad
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
color = pipeline.create(dai.node.ColorCamera)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutColor = pipeline.create(dai.node.XLinkOut)

# Configurar cámaras
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
color.setPreviewSize(640, 360)
color.setBoardSocket(dai.CameraBoardSocket.RGB)

# Configurar nodo de profundidad
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setSubpixel(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

# Configurar XLinkOut
xoutDepth.setStreamName("depth")
xoutColor.setStreamName("color")

# Enlazar nodos
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)
color.preview.link(xoutColor.input)

# Ejecutar pipeline
with dai.Device(pipeline, usb2Mode=True) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    colorQueue = device.getOutputQueue(name="color", maxSize=4, blocking=False)
    
    while True:
        depthFrame = depthQueue.get().getFrame()  # Obtener el frame de profundidad
        colorFrame = colorQueue.get().getCvFrame()  # Obtener el frame de color

        # Realizar predicciones con YOLOv8 en el frame de color
        results = model.predict(colorFrame, conf=0.65)
        annotated_frame = results[0].plot()

        # Resetear las listas de detecciones por clase en cada frame
        boxes_data = []

        # Procesar detecciones
        for detection in results[0].boxes:
            class_id = int(detection.cls[0])
            class_name = model.names[class_id]
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            theta_y = (anguloFovMed/180)*(centroid_y-180)
            theta_x = (anguloFovMed2/320)*(centroid_x-320)

            # Obtener el valor de profundidad en el centroide
            if 0 <= centroid_x < depthFrame.shape[1] and 0 <= centroid_y < depthFrame.shape[0]:
                z_value = depthFrame[centroid_y, centroid_x]/1000
                if class_name == 'ABB':
                    h = z_value / 1000
                else:
                    h = 2.4
                x = h * np.tan(np.deg2rad(theta_x)) * (1 / 0.8887) - 0.0102
                x = -x
                y = h * np.tan(np.deg2rad(theta_y))

                # Dibujar un círculo en el centroide y mostrar la distancia en Z
                cv2.circle(annotated_frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
                cv2.putText(annotated_frame, f"Z: {round(z_value, 2)} m", (centroid_x + 10, centroid_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                cv2.putText(annotated_frame, f"X: {round(x, 2)} m", (centroid_x + 10, centroid_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                cv2.putText(annotated_frame, f"Y: {round(y, 2)} m", (centroid_x + 10, centroid_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

                cv2.line(annotated_frame, (320, 0), (320, 360), (0, 0, 255), 2)
                cv2.line(annotated_frame, (0, 180), (640, 180), (0, 0, 255), 2)
                # Almacenar la detección en la lista adecuada
                detection_info = {"class": class_name, "x": x, "y": y, "z": z_value}
                if class_name == 'BOX':
                    boxes_data.append(detection_info)

        # Enviar coordenadas de la primera caja detectada al cliente MATLAB
        if boxes_data:
            first_box = boxes_data[0]
            first_box['x']=first_box['x']*1000
            first_box['y']=first_box['y']*1000

            message = f"{first_box['x']},{first_box['y']},330\n"
            conn.send(message.encode('utf-8'))
            print('Enviado:', message.strip())

        # Mostrar el frame anotado
        cv2.imshow("YOLOv8 Inference with Depth", annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

conn.close()
server_socket.close()
