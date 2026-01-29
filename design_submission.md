# Intelligent Vision System for Smart Manufacturing Safety & Analytics

**Assignment Submission for Computer Vision / Deep Learning Engineer Role**

---

## 1. Objective
To design a robust, real-time multi-camera vision system for a **Smart Manufacturing Factory**. The system aims to enhance worker safety, optimize logistics, and monitor production quality using edge AI.

**Key Goals:**
-   **Safety:** Detect workers entering hazardous zones and ensure PPE compliance.
-   **Logistics:** Track forklifts and count pallets.
-   **Quality:** Anomaly detection on conveyor belts.
-   **Situational Awareness:** Panoramic stitching of large warehouse floors.

---

## 2. Hardware & Platform Selection

**Selected Edge Compute:** **NVIDIA Jetson AGX Orin (32GB/64GB)**

*   **Justification:**
    *   **Compute:** Delivers up to 275 TOPS, essential for running multiple concurrent models (Detection, Segmentation, Stitching) across 4-8 camera streams.
    *   **Memory:** Unified memory architecture allows zero-copy efficiency between CPU and GPU, critical for high-bandwidth video buffers.
    *   **IO:** Supports multiple GMSL2 or Ethernet cameras via virtual channels.
    *   **Thermals/Power:** Configurable power modes (15W - 60W) fit industrial power envelopes better than x86 servers.

**Alternative (Low Cost):** **NVIDIA Jetson Orin NX (16GB)** for smaller, cell-based deployments.

---

## 3. System Architecture Design

### High-Level Data Flow

```mermaid
graph TD
    subgraph "Sensing Layer"
        Cam1[Camera 1 (RTSP/GMSL)]
        Cam2[Camera 2 (RTSP/GMSL)]
        Cam3[Camera 3 (RTSP/GMSL)]
    end

    subgraph "Edge Compute (Jetson Orin) - DeepStream Pipeline"
        direction TB
        Input[NvStreamMux\n(Batching & Synchronization)]
        
        PreProc[Preprocessing\n(Resize, Color Conv, undistort)]
        
        Infer1[PGIE: Object Detection\n(YOLOv8 - TensorRT)]
        Tracker[NvTracker\n(DeepSORT / NvDCF)]
        
        Infer2[SGIE: Segmentation\n(SegFormer - TensorRT)]
        Infer3[SGIE: Anomalies\n(Autoencoder)]
        
        Analytics[NvDsAnalytics\n(Line Crossing, ROI Counting)]
        
        Stitch[CUDA/VPI Image Stitching\n(Bird's Eye View)]
        
        MsgConv[NvMsgConv\n(JSON Payload Gen)]
        MsgBroker[NvMsgBroker\n(Kafka Producer)]
        
        OSD[On-Screen Display\n(Local Alerting)]
    end

    subgraph "Communication Layer"
        Kafka[Kafka Broker / MQTT Bus]
    end

    subgraph "Cloud / On-Prem Backend"
        Storage[Time-Series DB & Video Archivist]
        Dashboard[Monitoring Dashboard]
        Retrain[Model Retraining Loop\n(TLT / TAO)]
    end

    Cam1 --> Input
    Cam2 --> Input
    Cam3 --> Input
    
    Input --> PreProc --> Infer1 --> Tracker
    Tracker --> Infer2
    Tracker --> Infer3
    
    Infer1 & Infer2 & Infer3 --> Analytics
    Analytics --> OSD
    Analytics --> MsgConv
    
    Cam1 & Cam2 & Cam3 -.-> Stitch --> OSD
    
    MsgConv --> MsgBroker --> Kafka --> Dashboard
    Kafka --> Storage
```

### Component Responsibilities

1.  **Edge (Jetson):**
    *   **Latency-Sensitive:** Inference, Object Tracking, Immediate Safety Alerts (GPIO triggers for buzzers), Image Stitching.
    *   **Privacy:** Video streams are processed locally; only metadata and snippet events are uploaded.
2.  **Cloud / Backbone:**
    *   **Long-term Storage:** Historical data of counts and violations.
    *   **Heavy Compute:** Model training (TAO Toolkit), Federated aggregation (if finding cross-site patterns).
3.  **Failure Handling:**
    *   **Network Loss:** Edge buffers messages in a local Redis/SQLite queue and syncs when connection returns.
    *   **Sensor Fail:** System alerts on "Camera Loss" signal via V4L2/RTSP status; fallback to overlapping camera if available.

---

## 4. Model & AI Pipeline Design

We utilize the **NVIDIA TAO (Train-Adapt-Optimize)** workflow to speed up development.

| Capability | Model Architecture | Training Strategy | Edge Suitability |
| :--- | :--- | :--- | :--- |
| **1. Object Detection** | **YOLOv8** or **PeopleNet (ResNet34)** | Transfer Learning on COCO + Custom Factory Dataset. | High accuracy/FPS tradeoff. TensorRT optimization is mature. |
| **2. Semantic Segmentation** | **SegFormer** or **UNet** | Trained on Cityscapes + Industrial Floor markings (Drivable vs Walkable zones). | Lightweight encoders (MixTransformer) run well on Jetson DLA. |
| **3. Analytics** | **NvDsAnalytics** (Plugin) | Logic-based (Line crossing, Loitering). No training needed. | Extremely low overhead (CPU based). |
| **4. Stitching** | **Feature-based (ORB)** + **Seam Carving** | Use **NVIDIA VPI (Vision Programming Interface)** for hardware-accelerated warping. | VPI uses VIC (Video Image Compositor) hardware, saving GPU for inference. |
| **5. Anomalies** | **Autoencoder / GAN** | **Unsupervised:** Train on "Normal" conveyor belt operation. High reconstruction error = Anomaly. | Handles unknown defects without needing labeled defect data. |
| **6. Patterns** | **DBSCAN / K-Means** | Clustering trajectory coordinate points (Heatmaps) to find "desire paths" or bottleneck areas. | Can run periodically on metadata. |

---

## 5. Optimization & Deployment

### 1. TensorRT Optimization
*   **Precision:** Convert weights to **FP16** (safe default) or **INT8** (aggressive) using Post-Training Quantization (PTQ) with a calibration dataset.
*   **Layer Fusion:** TensorRT fuses vertical (Conv+Bias+Relu) and horizontal layers to reduce kernel launch overhead.

### 2. DeepStream Utilization
*   **Zero-Copy:** Use `NvBufSurface` to keep frames in GPU memory from capture → inference → display. No CPU copying.
*   **Batching:** `NvStreamMux` batches frames from 4 cameras into a single tensor for the GPU, maximizing throughput.

### 3. Containerized Deployment
*   **Docker:** Use `nvcr.io/nvidia/deepstream:6.3-triton` base images.
*   **OTA Updates:** Deploy new model weights via a Kubernetes agent (k3s) or AWS IoT Greengrass running on the Jetson.

---

## 6. Trade-off Analysis

| Trade-off | Decision | Reasoning |
| :--- | :--- | :--- |
| **Accuracy vs. Latency** | Prioritize **Latency** for Safety; **Accuracy** for Quality. | Safety alerts must be <100ms. We use smaller backbones (ResNet18/MobileNet) for detection, but heavier ones for defect analysis. |
| **Edge vs. Cloud** | **Edge-First**. | Video bandwidth (4x 4K streams) is too high/expensive for cloud upload. Privacy regulations (GDPR/worker monitoring) favor local processing. |
| **FP32 vs. INT8** | **INT8** for Detection. | 4x speedup with <1% mAP drop. Essential for running multi-model pipelines on limited power. |
| **3D vs. Stitching** | **Stitching (2D)**. | Full 3D reconstruction requires high overlap and static scenes. 2D Panoramic stitching is sufficient for warehouse situational awareness and cheaper on compute. |

---

## 7. Failure Scenarios

1.  **Hardware Crash / Overheat:**
    *   **Watchdog:** Hardware watchdog timer reboots system if OS freezes.
    *   **Throttling:** `nvpmodel` dynamic scaling reduces clock speeds if temps exceed 85°C to prevent shutdown.
2.  **Model Drift:**
    *   **Symptom:** False positives increase (e.g., new uniforms detected as violations).
    *   **Mitigation:** "Active Learning" loop. Edge uploads low-confidence images to cloud for human labeling, then OTA updates the model.
3.  **Network Outage:**
    *   The safety system (GPIO lights/buzzers) continues to work autonomously (Edge Logic).
    *   Analytics data is cached locally (Store-and-Forward).
