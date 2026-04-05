from ultralytics import YOLO

model = YOLO("../yolov8n.pt")

model.train(
    data="../datasets/detection/data.yaml",
    epochs=10,
    imgsz=416,
    batch=4,
    device="cpu"
)

model.val()