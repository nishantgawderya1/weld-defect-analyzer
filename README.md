# 🔬 Weld Defect Radiographic Inspector

A deep-learning-powered computer vision system designed for Non-Destructive Testing (NDT) and Quality Assurance (QA). It utilizes a customized **YOLOv8** classification model to analyze radiographic (X-ray) weld images, automatically identifying structural defects and providing actionable, standard-compliant engineering recommendations.

---

## 🚀 Key Features

* **Real-time Diagnostic Playground**: A bespoke drag-and-drop web dashboard to upload and analyze weld radiographs instantly.
* **Preloaded Test Samples**: High-end validation playground loaded with holdout evaluation samples of each defect category.
* **Standard-Compliant Action Protocols**: Incorporates physical defect definitions and response guidelines mapped to international standards (**ISO 5817**, **ASME BPVC Section V/VIII**, and **AWS D1.1**).
* **Interactive Model Performance Analytics**: Direct integration with the model's historical training logs, presenting interactive loss tracks, accuracy metrics, normalized confusion matrices, and validation batch previews.

---

## 🛠️ Defect Classification Standard

The system classifies weld anomalies into four core categories, matching their industrial quality severity:

| Class Code | Defect Name | Industrial Severity | Action Protocol |
| :--- | :--- | :---: | :--- |
| **CR** | **Crack** | 🔴 `CRITICAL FAILURE` | Reject weld immediately. Perform mechanical grinding, stress-relief heat treatment, or complete re-welding. |
| **LP** | **Lack of Penetration** | 🟡 `SERIOUS DEFECT` | Root bonding failure. Requires mechanical root grinding and re-welding. |
| **PO** | **Porosity** | 🟠 `WARNING / MODERATE` | Trapped gas cavities. Inspect shield gas flow and electrode moisture; grind and re-weld if pore density exceeds limits. |
| **ND** | **No Defect** | 🟢 `QUALITY PASS` | Structurally sound weld joint conforming to design specifications. |

---

## 📁 Project Structure

```text
├── data/                  # Workspace datasets
│   └── processed/         # YOLO-compatible structured dataset (train/val/test splits)
├── models/                # Trained standalone model exports
├── runs/                  # YOLO training & evaluation runs
│   └── classify/
│       └── runs/
│           └── classify/
│               └── weld_v1/     # Active model run (weights, confusion matrix, csv logs)
├── src/                   # Python Core Source
│   ├── app.py             # Streamlit Dashboard Web UI
│   ├── predict.py         # Modular YOLO Inference Engine & Metadata Parser
│   ├── prepare_dataset.py # Automated raw data migrator and class mapping utility
│   └── train.py           # Model training config script
├── requirements.txt       # Project dependencies (Torch, Ultralytics, Streamlit, etc.)
├── riawelc.yaml           # Dataset configuration and class map definitions
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.10+** and **Git** installed on your system.

### 2. Installation
Clone the repository and navigate into the project directory:
```powershell
git clone https://github.com/nishantgawderya1/weld-defect-analyzer.git
cd weld-defect-analyzer
```

### 3. Virtual Environment Setup
Initialize a Python virtual environment and install the required dependencies:
```powershell
# Create venv
python -m venv venv

# Activate venv (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Running the Dashboard
Launch the high-end Streamlit QA dashboard locally:
```powershell
streamlit run src/app.py
```
The application will open automatically in your browser at **[http://localhost:8501](http://localhost:8501)**.

---

## 📊 Model Training & Performance
The model was trained utilizing a **YOLOv8 Nano Classifier** (`yolov8n-cls`) for **50 epochs** with an input size of **224x224 pixels** and mini-batch sizes of **16**.

* **Model Weights**: Saved at `runs/classify/runs/classify/weld_v1/weights/best.pt`
* **Historical Logs**: Stored in `results.csv` and rendered dynamically under the **Performance Analytics** tab inside the dashboard.
