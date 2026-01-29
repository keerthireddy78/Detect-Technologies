# Edge AI Computer Vision – Proof of Concept

This repository contains a proof-of-concept implementation and system design
for a real-time multi-camera intelligent vision system, developed as part of
a hiring assignment for Detect Technologies.

The focus is on practical edge AI system design using lightweight deep learning
models, optimized for deployment on resource-constrained edge devices such as
NVIDIA Jetson.

---

##  Use Case

**Smart / Industrial Environment – Safety & Operational Analytics**

The system is designed to monitor structured environments such as industrial
sites, construction zones, transport hubs, and manufacturing surroundings where
people and vehicles operate in close proximity.

Key capabilities demonstrated:
- Person and vehicle detection
- Safety and compliance awareness (PPE context)
- Scene understanding via semantic segmentation
- Video analytics (counting, zone-based logic)
- Multi-camera spatial awareness using image stitching
- Pattern discovery using unsupervised clustering

---

##  AI Capabilities Demonstrated

- **Object Detection:** YOLOv8 (lightweight variant)
- **Semantic Segmentation:** DeepLabV3
- **Video Analytics:** Line-crossing and zone-based logic
- **Image Stitching:** Homography-based panoramic view
- **Clustering:** KMeans-based activity hotspot detection

This repository focuses on **functional validation and system design**, not
full-scale training or deployment.

---

## 📊 Dataset Usage

Publicly available datasets and images are used for proof-of-concept validation:

- **PPE Dataset (YOLOv8 format)**  
  https://www.kaggle.com/datasets/shlokraval/ppe-dataset-yolov8

- Additional publicly available images from industrial/construction and
  structured traffic environments are used to demonstrate generalization.

These datasets capture safety-critical interactions between people, equipment,
and vehicles, which closely resemble real-world industrial and manufacturing
scenarios.

For production deployment, models are expected to be fine-tuned using
site-specific industrial data via **NVIDIA Transfer Learning Toolkit (TLT)**.

---

##  System Design & Architecture

The system follows an **edge-first architecture**:
- All latency-sensitive tasks (inference, tracking, analytics) run on the edge
- Cloud is used only for retraining, long-term analytics, and visualization
- Designed to operate reliably under limited compute, power, and network
  constraints

Detailed architecture, trade-off analysis, and deployment strategy are provided
in the design document.

---

## Repository Structure

├── cv_poc_notebook.ipynb # Core AI capability demonstrations here i used a bus image in traffic
├── yolo.ipynb # YOLO experimentation notebook
├── design_submission.md # System architecture & design document
├── requirements.txt # Python dependencies
├── bus.jpg # Sample structured environment image that i used in poc_notebook
└── README.md # Project overview

The `yolo.ipynb` notebook provides a focused demonstration of YOLOv8-based
object detection and is used to validate detection performance independently
before integration into the full pipeline.


---

## Setup & Execution

Install dependencies:
```bash
pip install -r requirements.txt

jupyter notebook cv_poc_notebook.ipynb


