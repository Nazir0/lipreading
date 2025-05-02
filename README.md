

# Lipreading

## Deep Learning for Visual Speech Recognition (Lipreading) - Team 18

### Project Overview

This project explores the application of deep learning techniques for visual speech recognition (lipreading) at the word level. The goal is to develop a system capable of identifying spoken words solely based on the visual information of a speaker's lip movements, without relying on audio input.

### Motivation

Traditional speech recognition systems struggle in noisy environments or situations where audio is unavailable. Furthermore, individuals with hearing impairments face significant communication barriers. Lipreading technology offers a potential solution to enhance communication accessibility and inclusivity in various scenarios, from live events and online meetings to educational resources.

### Dataset

The primary dataset used is the **Lip Reading in the Wild (LRW)** dataset.

* Contains short video clips (1.16s, 29 frames @ 25fps) from BBC news footage.
* Features hundreds of speakers and 500 distinct English words.

For this phase:

* Focused on **60 target words**.
* Videos were re-partitioned: **Training (80%)**, **Validation (10%)**, **Testing (10%)** using stratified sampling.

---

## Methodology

### Preprocessing

* Raw video frames are processed to isolate the mouth region.
* [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html) is used for facial landmark detection.
* The outer lip contour is identified and cropped with a margin (0.3).
* Region is resized to `64x64` pixels and converted to **grayscale**.
* Exactly **29 frames** per clip → input shape: `(29, 64, 64, 1)`.

### Model Architecture

A hybrid neural network combining spatial and temporal features:

* **3D Convolutional Frontend**: Processes initial frame sequence.
* **TimeDistributed MobileNetV3Small**: Extracts per-frame spatial features.
* **Transformer Encoder (4 blocks, 8 heads)**: Models temporal dependencies.
* **Positional Embeddings**: Inject sequence order into the Transformer.
* **Attention Pooling**: Aggregates temporal features.
* **Dense Classifier**: Outputs prediction (softmax over 60 classes).

### Training

* Framework: **TensorFlow/Keras**
* Hardware: **Google Colab (T4 GPU)**
* Optimizer: `AdamW` with cosine decay
* Loss: `SparseCategoricalCrossentropy`
* Mixed-precision training enabled
* **Early Stopping** on validation accuracy

---

## Results

* Achieved **70–73% test accuracy** on the 60-word task.
* Some confusion noted between visually similar words (e.g., *WITHIN* / *WITHOUT*).

---

## Demonstration App

A **Flask** web application allows testing via:

* Curated sample videos
* Live webcam input

---

## Getting the Dataset (LRW)

1. Visit the University of Oxford VGG LRW Dataset page.
2. Request access (release form).
3. Upon approval, download `.tar` archives.

> 📝 Note: For the demo, only specific video files in `SAMPLE_VIDEOS_CONFIG` are needed.

---

## Setup Instructions

### Prerequisites

* Python 3.8+
* pip

### Steps

1. **Clone the Repository** (if applicable):

```bash
git clone <repository_url>
cd <repository_directory>
```

2. **Create and Activate Virtual Environment**:

```bash
python -m venv venv
```

**macOS/Linux**:

```bash
source venv/bin/activate
```

**Windows**:

```bash
.\venv\Scripts\activate
```

3. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

Make sure `requirements.txt` contains:
`Flask`, `tensorflow`, `opencv-python`, `mediapipe`, `numpy`, `Werkzeug`.

4. **Prepare Required Files**:

#### Model Weights

* Create folder: `model_files/`
* Add your best model:
  `ckpt-epoch_55-val_acc_0.700.weights.h5`
* Set correct filename in `app.py` under `BEST_CHECKPOINT_FILENAME`.

#### Class Map

* File: `lrw_60words_class_to_int_map.json` → in `model_files/`
* Set `CLASS_MAP_FILENAME` in `app.py`.

#### Sample Videos

* Folder structure:

```
static/
└── videos/
    ├── BECAUSE/
    │   └── test/
    │       └── BECAUSE_00003_h264.mp4
    ├── BETWEEN/
    │   └── test/
    │       └── BETWEEN_00023_h264.mp4
    └── ...
```

Ensure paths match `SAMPLE_VIDEOS_CONFIG` in `app.py`.

---

## Running the Demo Application

1. **Activate virtual environment**
2. **Navigate to project directory**
3. **Run app**:

```bash
python app.py
```

4. **Access**:

Open a browser at:

* `http://127.0.0.1:5001`
* Or from other devices via local IP (e.g., `http://192.168.1.100:5001`)

---

## Using the Demo Application

### 1. Sample Video Prediction

* Choose a word from the dropdown.
* Click **"Run Lipreading"**
* See:

  * Filename
  * Ground Truth
  * Predicted Word
  * Confidence Score
  * Word Mapping Panel

### 2. Live Webcam Snippet

* Click **Start Webcam**
* Position face in frame
* Click **"Record Snippet (\~1.5s)"**
* After processing, view predicted word and confidence score

---

## Code Structure

```
├── app.py
├── requirements.txt
├── model_files/
│   ├── ckpt-epoch_55-val_acc_0.700.weights.h5
│   └── lrw_60words_class_to_int_map.json
├── static/
│   └── videos/
│       ├── BECAUSE/test/BECAUSE_00003_h264.mp4
│       ├── BETWEEN/test/BETWEEN_00023_h264.mp4
│       └── ...
├── templates/
│   ├── index.html
│   └── result.html
└── uploads/
```

---

## Future Work

* Train on the full **500-word** LRW dataset.
* Extend to **sentence-level** real-time lipreading.
* Optimize model for edge devices (e.g., quantization, pruning).
* Combine with language models for better semantic output.

---

## Team

* **Aman Shah**
* **Mouhamed Nazir Mbow**
* **Jalal Cherkaoui**
* **Anwar Benhnini**
* **Mohamed Yassir Ousdid**

---

