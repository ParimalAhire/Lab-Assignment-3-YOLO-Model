from ultralytics import YOLO

model = YOLO("../yolov8n-obb.pt")

model.train(
    data="../datasets/obb/data.yaml",
    epochs=10,
    imgsz=416,
    batch=4,
    device="cpu"
)

model.val()