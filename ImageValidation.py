# from ultralytics import YOLO
# import cv2

# model=YOLO("runs/detect/train/weights/best.pt")

# image=cv2.imread("imagen800.jpg")

# resultados= model.predict(image)

# anotaciones=resultados[0].plot()

# cv2.imshow("imagen",anotaciones)
############################################################################################################
# Run inference on the source
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('runs/detect/train/weights/best.pt')

# Run inference on 'bus.jpg' with arguments
model.predict('image_90.png', save=True, imgsz=640,show=True)