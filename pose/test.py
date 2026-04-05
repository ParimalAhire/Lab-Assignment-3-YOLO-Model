from ultralytics import YOLO

model = YOLO("models/pose.pt")

results = model.predict(
    source="test_images/pose",
    imgsz=416,
    save=True
)

print("Pose testing done")