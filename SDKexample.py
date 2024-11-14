#!/usr/bin/env python3

import cv2
import depthai as dai

# Coordenadas del píxel específico (ajusta según tus necesidades)
PIXEL_X, PIXEL_Y = 160, 120
ROI_SIZE = 5  # Tamaño de la región de interés

# Crear pipeline
pipeline = dai.Pipeline()

# Crear nodos
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)


spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)

# Configurar propiedades de las cámaras
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Configurar propiedades del nodo de profundidad
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setSubpixel(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

# Configurar Spatial Location Calculator
spatialLocationCalculator.setWaitForConfigInput(False)
spatialLocationCalculator.inputDepth.setBlocking(False)
spatialLocationCalculator.inputDepth.setQueueSize(1)

# Crear la región de interés (ROI) usando SpatialLocationCalculatorConfigData
config = dai.SpatialLocationCalculatorConfigData()
config.roi = dai.Rect(
    (PIXEL_X - ROI_SIZE) / monoLeft.getResolutionWidth(),
    (PIXEL_Y - ROI_SIZE) / monoLeft.getResolutionHeight(),
    2 * ROI_SIZE / monoLeft.getResolutionWidth(),
    2 * ROI_SIZE / monoLeft.getResolutionHeight(),
)
spatialLocationCalculator.initialConfig.addROI(config)

# Configurar XLinkOut
xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")

# Enlazar nodos
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(spatialLocationCalculator.inputDepth)
spatialLocationCalculator.out.link(xoutSpatialData.input)
stereo.depth.link(xoutDepth.input)

# Ejecutar pipeline
with dai.Device(pipeline, usb2Mode=True) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)

    while True:
        # Obtener el frame de profundidad
        depthFrame = depthQueue.get().getFrame()
        depthFrameColor = cv2.applyColorMap(cv2.convertScaleAbs(depthFrame, alpha=0.03), cv2.COLORMAP_HOT)

        # Obtener coordenadas espaciales del píxel
        spatialData = spatialQueue.get().getSpatialLocations()

        if spatialData:
            locationData = spatialData[0].spatialCoordinates
            x_value = locationData.x / 1000.0  # Convertir a metros
            y_value = locationData.y / 1000.0
            z_value = locationData.z / 1000.0

            # Mostrar las coordenadas en la imagen
            cv2.putText(depthFrameColor, f"X: {x_value:.2f} m", (PIXEL_X + 10, PIXEL_Y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.putText(depthFrameColor, f"Y: {y_value:.2f} m", (PIXEL_X + 10, PIXEL_Y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.putText(depthFrameColor, f"Z: {z_value:.2f} m", (PIXEL_X + 10, PIXEL_Y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            # Dibujar un círculo en el píxel específico
            cv2.circle(depthFrameColor, (PIXEL_X, PIXEL_Y), 5, (255, 255, 255), -1)

        # Mostrar el frame anotado
        cv2.imshow("depth", depthFrameColor)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
