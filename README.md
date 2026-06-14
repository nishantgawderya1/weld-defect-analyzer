<div align="center">

# 🔬 Weld Defect Radiographic Inspector

### AI-Powered Non-Destructive Testing for Industrial Weld Quality Assurance

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-0B23A9?style=for-the-badge&logo=yolo&logoColor=white" alt="YOLOv8">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <a href="https://huggingface.co/spaces/nishant-online-cf/weld-inspection-ai">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-FFD21E?style=for-the-badge" alt="Hugging Face Spaces">
  </a>
</p>

**[🚀 Live Demo](https://huggingface.co/spaces/nishant-online-cf/weld-inspection-ai)** &nbsp;·&nbsp; **[⚙️ Getting Started](#-getting-started)** &nbsp;·&nbsp; **[🧠 Model & Performance](#-model-training--performance)**

</div>

---

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

## 🌐 Deploying to the Web (Streamlit Community Cloud)

Put the dashboard online for free in a few minutes. No server or DevOps knowledge needed.

### Step 1 — Push your code to GitHub
Make sure your latest changes (including the trained model weights) are on the `main` branch:
```powershell
git add -A
git push origin main
```
> The model file `best.pt` is normally ignored by Git. It has already been **force-added** so the cloud app can load it. If you train a new model, add it the same way:
> `git add -f runs/classify/<run_name>/weights/best.pt`

### Step 2 — Create the app
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in:
   * **Repository:** `nishantgawderya1/weld-defect-analyzer`
   * **Branch:** `main`
   * **Main file path:** `src/app.py`

### Step 3 — ⚠️ Set the Python version (most important!)
Before clicking deploy, open **Advanced settings** and set:

* **Python version → `3.12`**

> **Why this matters:** This project uses `torch` and `ultralytics`, which **do not have installers for Python 3.13 or 3.14 yet**. If you leave the default (newer) version, the build fails with `No matching distribution found`. Python **3.12** is the safe choice.

### Step 4 — Deploy
Click **Deploy**. The first build takes a few minutes (it downloads PyTorch). When it finishes, your app is live at a public `*.streamlit.app` URL. 🎉

### Updating the live app
Just `git push` to `main` — Streamlit redeploys automatically. If the app misbehaves after an update, open **Manage app → ⋮ → Reboot app** for a clean restart.

---

## 🩹 Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `No matching distribution found for pywin32` / build fails resolving dependencies | `requirements.txt` had Windows-only / unneeded packages | Already fixed — `requirements.txt` is now a minimal Linux-friendly list. Keep it lean; don't paste a full `pip freeze` into it. |
| `Could not find a version that satisfies torch` | Cloud is using Python **3.13/3.14** (no PyTorch wheels) | Set **Python version → 3.12** in Advanced settings, then reboot. |
| `WeldPredictor.__init__() got an unexpected keyword argument 'apply_preprocessing'` | The app is running **old code still cached in memory** (Streamlit reruns `app.py` but does not reload imported modules) | Fully **restart** the process: locally press `Ctrl+C` and run `streamlit run src/app.py` again; on Cloud use **Reboot app**. A browser "Rerun" is not enough. |
| `Model predictor not initialized` / `best.pt not found` | The model weights aren't in the deployed repo | Force-add the weights and push: `git add -f runs/classify/<run_name>/weights/best.pt`. |
| App runs but crashes when loading a model | Free tier memory limit (~1 GB); PyTorch + YOLO is heavy | Reboot; if it persists, stick to the lightweight `yolov8n-cls` model (already the default). |

---

## 📊 Model Training & Performance
The model was trained utilizing a **YOLOv8 Nano Classifier** (`yolov8n-cls`) for **50 epochs** with an input size of **224x224 pixels** and mini-batch sizes of **16**.

* **Model Weights**: Saved at `runs/classify/runs/classify/weld_v1/weights/best.pt`
* **Historical Logs**: Stored in `results.csv` and rendered dynamically under the **Performance Analytics** tab inside the dashboard.
