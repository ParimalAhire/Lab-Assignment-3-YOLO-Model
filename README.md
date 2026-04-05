# Lab Assignment 3 – YOLO Model  
## Multi-Task Vision System using YOLOv8

---

## Course Details

**Course Name:** Deep Learning  
**Lab Assignment:** Lab Assignment 3  
**Semester:** VI  
**Topic:** Multi-Task Vision System using YOLOv8  

---

## Group Details

| Student Name | PRN |
|--------------|-------------------|
| Parimal Ahire | 202301040067 |
| Atharva Suryawanshi | 202301040283 |
| Rajveersinh Kher | 202301040233 |
| Mohit Patil | 202301040272 |

---

## Objective

The objective of this lab assignment is to design and implement a multi-task computer vision system using modern YOLOv8 models. The system demonstrates how a single framework can perform multiple vision tasks efficiently using lightweight nano models and local deployment.

The following tasks are implemented:

- Object Detection  
- Image Classification  
- Pose Estimation  
- Oriented Bounding Box (OBB) Detection  
- Local deployment using YOLOv8 nano models  

---

## Tools and Technologies Used

**Programming Language:**  
Python 3.11  

**Framework:**  
Ultralytics YOLOv8  

**Libraries:**

- PyTorch  
- OpenCV  
- Matplotlib  
- NumPy  
- Pillow  

**Dataset Source:**  
Roboflow

---

## System Configuration

**Processor:** Intel Core i5-1335U  
**GPU:** Not used (CPU based training)  
**RAM:** 16 GB  
**Storage:** SSD  
**Operating System:** Windows  
**Development Environment:** VS Code  
**Python Version:** 3.11  

---

## Project Structure

```
Lab-Assignment-3-YOLO-Model/

├── detection/
├── classification/
├── pose/
├── obb/

├── datasets/
│   ├── detection/
│   ├── classification/
│   ├── pose/
│   └── obb/

├── models/

├── test_images/
│   ├── detection/
│   ├── classification/
│   ├── pose/
│   └── obb/

├── runs/
├── results/

├── deploy.py
├── requirements.txt
├── README.md
└── report.pdf
```

## Dataset Details

All datasets were obtained from Roboflow and exported in YOLO compatible format.

### Object Detection Dataset  
Helmet Detection Dataset  
https://universe.roboflow.com/humanhead-azp04/helmet-detection-x2yqr

### Classification Dataset  
Animal Classification Dataset  
https://universe.roboflow.com/detection-aq67m/animal-classification-bckhx

### Pose Estimation Dataset  
Human Pose Dataset  
https://universe.roboflow.com/custom-wyfc3/human-pose-3vmzj

### OBB Dataset  
PCB Component Detection Dataset  
https://universe.roboflow.com/hieu2959/obb-blkiy

Dataset preprocessing included resizing, normalization, and YOLO format annotation verification.

---

## Implementation

### Part A – Object Detection

**Model:** YOLOv8n  
**Epochs:** 10  
**Image Size:** 416 × 416  

The model detects helmets in images using bounding boxes and confidence scores.

---

### Part B – Image Classification

**Model:** YOLOv8n-cls  
**Epochs:** 8  
**Image Size:** 224 × 224  

The model classifies animals into their respective categories using transfer learning.

---

### Part C – Pose Estimation

**Model:** YOLOv8n-pose  
**Epochs:** 10  
**Image Size:** 416 × 416  

The model detects human body keypoints and generates pose skeleton visualizations.

---

### Part D – Oriented Bounding Box Detection

**Model:** YOLOv8n-obb  
**Epochs:** 10  
**Image Size:** 416 × 416  

The model detects rotated bounding boxes for PCB components.

---

## Deployment

Deployment is implemented locally using a Python script (`deploy.py`).

The deployment demonstrates:

- Model selection  
- Image selection using file picker  
- Local inference  
- Result visualization  
- Execution time measurement  
- Automatic result saving  

### To run deployment:

```
pip install -r requirements.txt
python deploy.py
```

This demonstrates lightweight YOLO model deployment on a local machine.

---

## Results

The system successfully performs all required tasks.

**Object Detection:**  
Helmet detection with bounding boxes and confidence scores.

**Classification:**  
Accurate classification of animals into categories.

**Pose Estimation:**  
Human pose detection with skeleton visualization.

**OBB Detection:**  
Rotated bounding box detection of PCB components.

Training graphs and prediction outputs are available inside the `runs` folder.

---

## Demo Video (Assignment Recording)

[Assignment Demo Video](https://drive.google.com/file/d/1EfxO9io0nMiHYYdk11jWPqXCT3_5aWXN/view?usp=drive_link)

---


## How to Run the Project

### Install required libraries:

```
pip install -r requirements.txt
```

### Run deployment:

```
python deploy.py
```

### Run individual training modules:

**Detection**

```
python detection/train.py
```

**Classification**

```
python classification/train.py
```

**Pose**

```
python pose/train.py
```

**OBB**

```
python obb/train.py
```

---

## Key Features

- Multi-task vision system  
- Lightweight YOLOv8 nano models  
- CPU based training  
- Roboflow datasets  
- Local deployment  
- Organized modular project structure  

---

## Learning Outcomes

Through this project the following concepts were understood:

- Transfer learning using pretrained models  
- YOLOv8 architecture and workflow  
- Computer vision pipelines  
- Model training and validation  
- Image preprocessing and annotation formats  
- Model deployment techniques  

---

## Conclusion

This lab assignment successfully demonstrates the implementation of YOLOv8 models for multiple computer vision tasks. The nano models provide efficient performance while maintaining lightweight deployment capability. The project highlights practical applications of deep learning in real-world computer vision problems.
