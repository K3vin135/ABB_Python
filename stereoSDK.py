import cv2
from ultralytics import YOLO
import depthai as dai
import numpy as np

# Listas para almacenar los datos de detecciones por clase
boxes_data = []
abb_data = []
abb_base_data = []

# Cargar el modelo YOLOv8
model = YOLO('runs/detect/train/weights/best.pt')

# Crear pipeline
pipeline = dai.Pipeline()

# Crear nodos de cámara y profundidad
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
color = pipeline.create(dai.node.ColorCamera)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutColor = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)

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

# Configurar Spatial Location Calculator
spatialLocationCalculator.setWaitForConfigInput(False)
spatialLocationCalculator.inputDepth.setBlocking(False)
spatialLocationCalculator.inputDepth.setQueueSize(1)

# Configurar XLinkOut
xoutDepth.setStreamName("depth")
xoutColor.setStreamName("color")
xoutSpatialData.setStreamName("spatialData")

# Enlazar nodos
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(spatialLocationCalculator.inputDepth)
stereo.depth.link(xoutDepth.input)
color.preview.link(xoutColor.input)
spatialLocationCalculator.out.link(xoutSpatialData.input)

# Ejecutar pipeline
with dai.Device(pipeline, usb2Mode=True) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    colorQueue = device.getOutputQueue(name="color", maxSize=4, blocking=False)
    spatialQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialConfigQueue = device.getInputQueue(name="spatialConfig", maxSize=4, blocking=False)

    while True:
        depthFrame = depthQueue.get().getFrame()  # Obtener el frame de profundidad
        colorFrame = colorQueue.get().getCvFrame()  # Obtener el frame de color

        # Realizar predicciones con YOLOv8 en el frame de color
        results = model.predict(colorFrame, conf=0.65)
        annotated_frame = results[0].plot()

        # Resetear las listas de detecciones por clase en cada frame
        boxes_data.clear()
        abb_data.clear()
        abb_base_data.clear()

        # Configurar ROI para cada detección en SpatialLocationCalculator
        config = dai.SpatialLocationCalculatorConfig()
        for detection in results[0].boxes:
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
            roi = dai.Rect(x1 / color.getResolutionWidth(), y1 / color.getResolutionHeight(),
                           (x2 - x1) / color.getResolutionWidth(), (y2 - y1) / color.getResolutionHeight())
            spatialConfigData = dai.SpatialLocationCalculatorConfigData()
            spatialConfigData.roi = roi
            config.addROI(spatialConfigData)

        # Enviar configuración al nodo de cálculo espacial
        spatialConfigQueue.send(config)

        # Obtener datos espaciales
        spatialData = spatialQueue.get().getSpatialLocations()
        for i, detection in enumerate(results[0].boxes):
            class_id = int(detection.cls[0])
            class_name = model.names[class_id]
            locationData = spatialData[i].spatialCoordinates
            x_value = locationData.x / 1000.0  # Convertir a metros
            y_value = locationData.y / 1000.0
            z_value = locationData.z / 1000.0

            # Dibujar un círculo en el centroide y mostrar la distancia
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            cv2.circle(annotated_frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
            cv2.putText(annotated_frame, f"X: {x_value:.2f} m", (centroid_x + 10, centroid_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            cv2.putText(annotated_frame, f"Y: {y_value:.2f} m", (centroid_x + 10, centroid_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            cv2.putText(annotated_frame, f"Z: {z_value:.2f} m", (centroid_x + 10, centroid_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            # Almacenar la detección en la lista adecuada
            detection_info = {"class": class_name, "x": x_value, "y": y_value, "z": z_value}
            if class_name == 'BOX':
                boxes_data.append(detection_info)
            elif class_name == 'ABB':
                abb_data.append(detection_info)
            elif class_name == 'ABB_BASE':
                abb_base_data.append(detection_info)

        # Imprimir las listas de detecciones después del procesamiento
        print("Boxes data:", boxes_data)
        print("ABB data:", abb_data)

        # Mostrar el frame anotado
        cv2.imshow("YOLOv8 Inference with Depth", annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
