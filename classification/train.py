from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="datasets/classification",
    epochs=8,
    imgsz=224,
    batch=8,
    device="cpu"
)

model.val()