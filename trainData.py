from ultralytics import YOLO
import cv2

def train_model():
    # Load a model
    model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data='dataset.yaml', epochs=45, imgsz=640)
    return results

if __name__ == '__main__':
    results = train_model()
