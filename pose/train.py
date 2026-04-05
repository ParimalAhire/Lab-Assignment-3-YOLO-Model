from ultralytics import YOLO

model = YOLO("../yolov8n-pose.pt")

model.train(
    data="../datasets/pose/data.yaml",
    epochs=10,
    imgsz=416,
    batch=4,
    device="cpu"
)

model.val()