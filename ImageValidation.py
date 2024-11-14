# Run inference on the source
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('runs/detect/train/weights/best.pt')

# Run inference on 'bus.jpg' with arguments
model.predict('image_90.png', save=True, imgsz=640,show=True)