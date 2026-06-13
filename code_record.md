# Code Record - Weld Defect Analyzer

This document tracks the changes and progress made on the **Weld Defect Analyzer** project.

## Project Overview
The goal of this project is to analyze weld defects using deep learning computer vision models (specifically YOLOv8 classification) and provide interactive quality assurance tools for industrial radiographic inspections.

## Current Project Structure
- `data/`: Processed datasets with standard `train`, `val`, and `test` splits matching YOLO classification guidelines.
- `models/`: Placeholder for trained model standalone exports.
- `notebooks/`: For experimental research and prototyping.
- `src/`: Core source code.
    - `prepare_dataset.py`: Script that automated the migration of 24k raw images to the processed workspace.
    - `train.py`: Script setting up parameters to train the YOLOv8-cls model for 50 epochs.
    - `predict.py`: Core inference engine wrapping YOLO classification probabilities and mapping predictions to international compliance standards.
    - `app.py`: High-end Streamlit web dashboard for live NDT inspection and performance analytics.
- `requirements.txt`: Project dependencies (includes `torch`, `ultralytics`, `streamlit`, `pandas`, `opencv-python`, etc.).
- `riawelc.yaml`: Configuration metadata listing the 4 defect classes: `CR` (Crack), `LP` (Lack of Penetration), `PO` (Porosity), `ND` (No Defect).
- `README.md`: Complete, professional project setup and metallurgical reference documentation.

---

## Work History

### 2026-04-26
- **Dataset Identification**: Verified the location and structure of the dataset.
    - **Path**: `C:\Users\nisha\Downloads\DB - Copy`
    - **Structure**: Contains `training`, `validation`, and `testing` folders.
    - **Classes**: `Difetto1`, `Difetto2`, `Difetto4`, `NoDifetto`.
- **Terminal Error Resolution**: Fixed a `FileNotFoundError` and `UnicodeEncodeError` in the terminal.
    - Resolved path mismatch in `Downloads` directory.
    - Replaced unicode characters in script output to prevent Windows terminal crashes.
- **Dataset Preparation**: Created and executed `src/prepare_dataset.py`.
    - **Action**: Automated the migration of ~24,000 images from `Downloads` to the project's `data/processed/` directory.
    - **Class Mapping**:
        - `Difetto1` -> `CR` (Crack)
        - `Difetto2` -> `LP` (Lack of Penetration)
        - `Difetto4` -> `PO` (Porosity)
        - `NoDifetto` -> `ND` (No Defect)
    - **Split Mapping**: `training` -> `train`, `validation` -> `val`, `testing` -> `test`.
    - **Status**: Successfully processed all splits into YOLO-compatible structure.
- **Environment Setup**: Initialized project directory and structure.
- **Dependency Management**: Populated `requirements.txt` with necessary libraries for computer vision and ML lifecycle management.

### 2026-05-26 & 2026-05-27
- **Model Verification**: Verified completed classification model training run (`weld_v1`) producing weight outputs:
  * Best weights: `runs/classify/runs/classify/weld_v1/weights/best.pt`
  * Static evaluation assets: `confusion_matrix_normalized.png`, `results.csv`, `results.png`, and `val_batch0_pred.jpg`.
- **Modular Prediction Core (`src/predict.py`)**:
  * Implemented `WeldPredictor` class.
  * Formatted predictions to return custom color-coded alert labels, severity classifications, and custom engineering action recommendations linked to international quality codes (**ISO 5817**, **ASME**, and **AWS**).
- **Interactive Web Interface (`src/app.py`)**:
  * Created a multi-tab Streamlit dashboard themed around a premium industrial smart-factory slate layout.
  * **Playground Tab**: Drag-and-drop file uploader, dynamic validation test sample loader (automatically resolves and displays test set images for evaluation), custom CSS glowing severity alert cards, and inline progress indicators matching classification codes.
  * **Analytics Tab**: Parses training metrics from `results.csv` on the fly to plot dynamic loss tracks and validation accuracy lines alongside normalized confusion matrices and prediction sample boards.
  * **Methodology Tab**: Documented metallurgical definitions, physical causes of weld defects, and code reference compliance checklists.
- **Project Documentation (`README.md`)**:
  * Formulated full readme documentation covering architecture, defect thresholds, virtual environment instructions, and execution processes.
- **System Compatibility & Deployment**:
  * Overcame Windows System32 stub path blocking by explicitly configuring and executing Git from Program Files: `C:\Program Files\Git\cmd\git.exe`.
  * Staged, committed, and successfully pushed all active dashboard and prediction files to the remote repository: `https://github.com/nishantgawderya1/weld-defect-analyzer.git` on branch `main`.

### 2026-06-13
- **Explainability & Model Versioning (commit `b053518`)**:
  * Added **Grad-CAM XAI heatmaps** to `src/predict.py` (`pytorch-grad-cam`) for visual defect localization.
  * Added a **model-version selector** to `src/app.py` (Baseline `weld_v1` vs Preprocessed `weld_v2`).
- **Preprocessing Pipeline (`src/preprocess.py`)**:
  * Implemented **CLAHE** (local contrast) + **non-local-means denoising** to clean radiographic film grain.
  * Outputs a parallel dataset at `data/preprocessed/` mirroring `data/processed/`.
- **v2 Training Scripts**:
  * `src/train.py` — trains `weld_v2_preprocessed` (YOLOv8n-cls, 50 epochs) on the preprocessed set.
  * `src/resume_train.py` — resume helper (now superseded by the GPU workflow below).
- **Streamlit Cloud Deployment Fix**:
  * Replaced the Windows `pip freeze` `requirements.txt` (which carried `pywin32` + ~110 unused dev packages and broke Linux dependency resolution) with a **minimal, Linux-friendly set** of only the 9 packages the app imports; swapped `opencv-python` → `opencv-python-headless`.
  * **Action required in the Streamlit dashboard:** set Python version to **3.12** (torch/ultralytics have no 3.14 wheels) and reboot.

---

## Outstanding / What's Left

- **`weld_v2_preprocessed` is NOT yet trained** — this is the only real gap. The app's "Preprocessed Model (weld_v2)" selector errors until `runs/classify/weld_v2_preprocessed/weights/best.pt` exists.
  * **Decision (2026-06-13):** train on a **free GPU** (Colab/Kaggle T4) — ~1–2 h for 50 epochs vs ~71 h on this CPU. CPU resume was rejected (poor ROI + YOLO's resume flag tends to ignore a lowered epoch cap).
  * **How (preferred — Kaggle):** run `notebooks/train_weld_v2_kaggle.ipynb` — attach RIAWELC as a Kaggle Dataset (upload once, reused free; no Drive), GPU = On, Internet = On, run cells, download `weld_v2_best.zip` from the Output tab.
  * **How (alt — Colab):** run `notebooks/train_weld_v2_colab.ipynb` (zip `DB - Copy` → upload to Drive → run cells → download `weld_v2_best.zip`).
  * Either way, unzip to `runs/classify/weld_v2_preprocessed/weights/best.pt`.
- **Dataset is local-only** — `data/` and `runs/` are gitignored. Raw RIAWELC (24,407 imgs: 15,863 train / 6,101 val / 2,443 test) lives at `C:\Users\nisha\Downloads\DB - Copy`; `data/processed` in the repo holds only 4 committed test samples.
- **To deploy v2 on Streamlit Cloud**, force-add the weight (`git add -f .../weld_v2_preprocessed/weights/best.pt`) since `*.pt` is gitignored.
- **Known quirk:** training previously nested runs as `runs/classify/runs/classify/<name>`; the app's `load_predictor` checks both that and the clean `runs/classify/<name>` path.

---
*Last updated: 2026-06-13*
