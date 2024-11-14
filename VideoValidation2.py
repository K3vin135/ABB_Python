import cv2
from ultralytics import YOLO
import depthai as dai
import numpy as np

# Listas para almacenar los datos de detecciones por clase
boxes_data = []
abb_data = []
abb_base_data = []

stepSize = 0.01
radius = 0.01  # Radio del círculo

newConfig = False

# Load the YOLOv8 model
model = YOLO('runs/detect/train/weights/best.pt')

# Crear pipeline para la cámara y la profundidad
pipeline = dai.Pipeline()

# Crear nodos de cámara y profundidad
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
color = pipeline.create(dai.node.ColorCamera)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

xoutColor = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

# Configurar cámaras
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Ajustar configuración de la cámara de color
color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
color.setIspScale(1, 3)  # 1920x1080 -> 640x360
color.setPreviewSize(640, 360)
color.setBoardSocket(dai.CameraBoardSocket.RGB)
color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
color.setInterleaved(False)

lrcheck = True  # Mejor manejo de oclusiones
subpixel = False  # Mejor precisión para distancias largas
extended_disparity = False  # Mayor profundidad mínima

# Configurar nodo de profundidad
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)
stereo.setExtendedDisparity(extended_disparity)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)  # Filtro mediano 7x7

spatialLocationCalculator.inputConfig.setWaitForMessage(False)

config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)

# Configurar XLinkOut
xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
xoutColor.setStreamName("color")

# Enlazar nodos
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
color.preview.link(xoutColor.input)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

# Ejecutar pipeline
with dai.Device(pipeline, usb2Mode=True) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    colorQueue = device.getOutputQueue(name="color", maxSize=4, blocking=False)

    while True:
        inDepth = depthQueue.get()  # Blocking call, will wait until a new data has arrived
        spatialData = spatialCalcQueue.get().getSpatialLocations()

        colorFrame = colorQueue.get().getCvFrame()  # Obtener el frame de color
        results = model.predict(colorFrame, conf=0.65)
        annotated_frame = results[0].plot()

        # Resetear las listas de detecciones por clase en cada frame
        boxes_data.clear()
        abb_data.clear()
        abb_base_data.clear()
        config = dai.SpatialLocationCalculatorConfig()

        # Procesar detecciones
        for detection in results[0].boxes:
            class_id = int(detection.cls[0])
            class_name = model.names[class_id]

            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            width = colorFrame.shape[1]
            height = colorFrame.shape[0]
            topLeft = dai.Point2f(centroid_x / width - stepSize, centroid_y / height - stepSize)
            bottomRight = dai.Point2f(centroid_x / width + stepSize, centroid_y / height + stepSize)

            roi = dai.Rect(topLeft, bottomRight)
            loc = dai.SpatialLocationCalculatorConfigData()
            loc.depthThresholds.lowerThreshold = 100
            loc.depthThresholds.upperThreshold = 10000
            loc.roi = roi
            config.addROI(loc)

        spatialCalcConfigInQueue.send(config)

        # Obtener los datos de profundidad y espacial
        inDepth = depthQueue.get()
        depthFrame = inDepth.getFrame()

        # Redimensionar el depthFrame para que coincida con el tamaño del colorFrame
        depthFrame = cv2.resize(depthFrame, (colorFrame.shape[1], colorFrame.shape[0]))

        # Convertir depthFrame a un formato de 8 bits
        depthFrameColor = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        spatialData = spatialCalcQueue.get().getSpatialLocations()

        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, (255, 255, 255))
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, (255, 255, 255))
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, (255, 255, 255))
            cv2.line(annotated_frame, (320, 0), (320, 360), (0, 0, 255), 2)
            cv2.line(annotated_frame, (0, 180), (640, 180), (0, 0, 255), 2)

        cv2.imshow("depth", depthFrameColor)
        print(depthFrame.shape)

        # Mostrar el frame anotado
        cv2.imshow("YOLOv8 Inference with Depth", annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
