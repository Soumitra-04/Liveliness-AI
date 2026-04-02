# 🚀 Liveliness-AI

**Liveliness-AI** is a modular, API-based multimodal deepfake detection system that analyzes **image, video, and audio** inputs to determine authenticity and generate a trust score with explainable insights.

---

## 🧠 Overview

With the rise of AI-generated media, detecting deepfakes has become critical for ensuring trust in digital content.
Liveliness-AI provides a **lightweight, explainable, and offline-capable solution** that processes multiple media types and evaluates their authenticity.

---

## ⚙️ Features

* 📤 **File Upload API** (FastAPI)
* 🗂️ **Local File Storage System**
* 🧾 **SQLite Metadata Tracking**
* 🧠 **Multimodal Analysis Engine**

  * 🖼️ Image Analysis (FFT-based)
  * 🎥 Video Analysis (MediaPipe + rPPG concepts)
  * 🔊 Audio Analysis (Librosa-based)
* 🧮 **Fusion Engine**

  * Combines outputs into a unified trust score
* 🔍 **Explainable Output**

  * Provides reasoning for authenticity score

---

## 🏗️ Project Structure

```
Liveliness-AI/
│
├── app/
│   ├── routes/        # API endpoints
│   ├── services/      # File handling, DB, type detection
│   ├── ai_engine/     # Image, video, audio processing + fusion
│   ├── models/        # Schemas & DB models
│   └── main.py        # FastAPI app entry
│
├── temp/              # Uploaded files
├── database/          # SQLite database
├── requirements.txt
├── run.py
└── README.md
```

---

## 🔄 Pipeline

```
Upload File
   ↓
API (/analyze)
   ↓
Save File (local storage)
   ↓
Detect File Type (image/video/audio)
   ↓
Store Metadata (SQLite)
   ↓
AI Processing (modality-specific)
   ↓
Fusion Engine
   ↓
Final Output (Score + Risk + Explanation)
```

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Soumitra-04/Liveliness-AI.git
cd Liveliness-AI
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the server

```bash
uvicorn app.main:app --reload
```

---

### 4. Open API docs

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Usage

* Use Swagger UI to upload:

  * `.jpg`, `.png` → Image
  * `.mp4`, `.avi` → Video
  * `.wav`, `.mp3` → Audio

---

## 📤 Sample Response

```json
{
  "authenticity_score": 42,
  "risk_classification": "HIGH",
  "flags": [
    "Facial landmark instability detected",
    "High-frequency anomalies in image",
    "Audio lacks natural variation"
  ]
}
```

---

## 🧠 Tech Stack

* **Backend:** FastAPI
* **Database:** SQLite
* **Image Processing:** Pillow, NumPy
* **Video Processing:** OpenCV, MediaPipe
* **Audio Processing:** Librosa
* **Computation:** SciPy

---

## 🏆 Key Highlights

* ✔ Fully offline (no external APIs required)
* ✔ Modular and scalable architecture
* ✔ Explainable AI outputs
* ✔ Real-time API-based system

---

## 🚧 Future Improvements

* Integration with deep learning models (ResNet, Wav2Vec)
* Browser extension for real-time verification
* Deployment on cloud (Docker + CI/CD)
* UI dashboard for visualization

---

## 👥 Team

* Soumitra Rajguru
* Team Members (add names)

---

## 📜 License

This project is for educational and hackathon purposes.

---

## 💡 Inspiration

Built to address the growing challenge of **misinformation, deepfakes, and digital trust** in modern systems.

---

⭐ If you like this project, consider giving it a star!


Hello this is Shubhan! 