import streamlit as st
import os
import pandas as pd
from PIL import Image
import numpy as np

# ── PAGE CONFIGURATION ──────────────────────────────────────────
st.set_page_config(
    page_title="Weld Defect Analyzer • Radiographic Inspection Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GUARDED PREDICTOR IMPORT (TEMPORARY DIAGNOSTIC) ─────────────
# Streamlit Cloud redacts uncaught exceptions, which hides the real reason
# `import cv2` fails inside ultralytics. Catching it here lets the page render
# the actual message plus an environment probe. Delete this block and restore
# the plain `from predict import WeldPredictor` once the deploy is healthy.
try:
    from predict import WeldPredictor
except Exception:
    import ctypes.util
    import importlib.metadata as md
    import platform
    import traceback

    st.error("Predictor import failed. Full diagnostic below — send this to debug the deploy.")

    st.markdown("#### Real traceback (unredacted)")
    st.code(traceback.format_exc(), language="text")

    st.markdown("#### Environment probe")
    probe = [f"python           : {platform.python_version()}"]

    opencv_dists = sorted(
        f"{d.metadata['Name']}=={d.version}"
        for d in md.distributions()
        if "opencv" in (d.metadata["Name"] or "").lower()
    )
    probe.append(f"opencv installed : {opencv_dists or 'NONE'}")
    probe.append(f"libGL found      : {ctypes.util.find_library('GL') or 'NOT FOUND (packages.txt did not apply)'}")

    for mod in ("numpy", "cv2", "torch", "ultralytics"):
        try:
            probe.append(f"{mod:17}: {__import__(mod).__version__} OK")
        except Exception as err:
            probe.append(f"{mod:17}: FAILED -> {type(err).__name__}: {err}")

    st.code("\n".join(probe), language="text")
    st.stop()

# ── BESPOKE MODERN CSS INJECTION ───────────────────────────────
st.markdown(
    """
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Header styling */
    .title-area {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 12px;
        border-bottom: 2px solid #334155;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .title-area h1 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        margin: 0 !important;
        font-size: 2.25rem !important;
        letter-spacing: -0.02em;
    }
    .title-area p {
        color: #94a3b8 !important;
        margin: 8px 0 0 0 !important;
        font-size: 1.05rem !important;
    }
    
    /* Sleek Custom Metric Cards */
    .custom-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 25px;
    }
    .custom-metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, border-color 0.2s;
    }
    .custom-metric-card:hover {
        transform: translateY(-2px);
        border-color: #475569;
    }
    .custom-metric-card .label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .custom-metric-card .value {
        font-size: 1.65rem;
        color: #f8fafc;
        font-weight: 700;
        margin-top: 6px;
    }
    .custom-metric-card .subtext {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Styled uploader area */
    div[data-testid="stFileUploader"] {
        background-color: #1e293b;
        border: 2px dashed #475569;
        border-radius: 12px;
        padding: 10px;
        transition: border-color 0.3s;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
    }
    
    /* Footer elements */
    .footer-bar {
        text-align: center;
        padding: 20px 0;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid #334155;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ── INITIALIZE MODEL ─────────────────────────────────────────────
@st.cache_resource
def load_predictor(model_version):
    try:
        if model_version == "Preprocessed Model (weld_v2)":
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            paths = [
                os.path.join(base_dir, "runs", "classify", "runs", "classify", "weld_v2_preprocessed", "weights", "best.pt"),
                os.path.join(base_dir, "runs", "classify", "weld_v2_preprocessed", "weights", "best.pt")
            ]
            model_path = None
            for p in paths:
                if os.path.exists(p):
                    model_path = p
                    break
            
            if model_path is None:
                st.sidebar.warning("⚠️ Preprocessed Model weights not found. Using Baseline.")
                return WeldPredictor()
            # v2 was trained on CLAHE+denoise images, so inputs must be preprocessed too.
            return WeldPredictor(model_path=model_path, apply_preprocessing=True)
        else:
            return WeldPredictor()
    except Exception as e:
        st.error(f"⚠️ Model Initialization Failed: {e}")
        return None

# ── HELPER: GET DYNAMIC TEST SAMPLE IMAGES ────────────────────────
def get_sample_image(class_code):
    """
    Finds the first available PNG image in data/processed/test/{class_code}/
    """
    base_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "test", class_code
    ))
    if os.path.exists(base_path):
        files = [f for f in os.listdir(base_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            return os.path.join(base_path, files[0])
    return None

# ── HEADER RENDER ────────────────────────────────────────────────
st.markdown(
    """
    <div class="title-area">
        <h1>🔬 Radiographic Weld Defect Inspector</h1>
        <p>Bespoke deep learning analysis of industrial weld radiographs. Detecting Cracks, Lack of Penetration, and Porosity anomalies.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ── SIDEBAR NAVIGATION ───────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <h3 style="color: #f8fafc; font-weight: 700; margin-bottom: 4px;">Weld Defect AI</h3>
        <p style="color: #64748b; font-size: 0.85rem; margin-bottom: 20px;">Precision Inspection System</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "NAVIGATION SYSTEM",
    ["🔬 Inspection Playground", "📊 Model Performance Analytics", "📖 Metallurgy & Methodology"],
    index=0
)
st.sidebar.markdown("---")

# Model selector in the sidebar
st.sidebar.markdown("### ⚙️ SYSTEM SETTINGS")
selected_model_ver = st.sidebar.selectbox(
    "Select Model Version",
    ["Baseline Model (weld_v1)", "Preprocessed Model (weld_v2)"]
)
st.sidebar.markdown("---")

predictor = load_predictor(selected_model_ver)

# Sidebar Info Card
model_display_name = "weld_v1 (Baseline)" if selected_model_ver == "Baseline Model (weld_v1)" else "weld_v2 (Preprocessed)"
st.sidebar.markdown(
    f"""
    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px; font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">
        <strong style="color: #f8fafc;">System Status:</strong> <span style="color: #10b981;">🟢 ACTIVE</span><br>
        <strong style="color: #f8fafc;">Hardware Target:</strong> PyTorch (CPU)<br>
        <strong style="color: #f8fafc;">Model Class:</strong> YOLOv8n-cls<br>
        <strong style="color: #f8fafc;">Active Model:</strong> {model_display_name}<br>
        <strong style="color: #f8fafc;">Version:</strong> {"v1.0.0" if selected_model_ver == "Baseline Model (weld_v1)" else "v1.1.0"}
    </div>
    """,
    unsafe_allow_html=True
)

# ── PAGE 1: INSPECTION PLAYGROUND ────────────────────────────────
if page == "🔬 Inspection Playground":
    st.markdown("### 🔬 Live Radiograph Analysis Panel")
    st.markdown("Upload a new weld X-ray image or load one of our industrial validation test samples to watch the deep learning model run real-time inference.")
    
    show_xai = False
    col_input, col_result = st.columns([2, 3], gap="large")
    
    with col_input:
        st.markdown("#### 📥 Image Source Selection")
        
        source_mode = st.radio(
            "Select Input Source",
            ["✨ Select Preloaded Industrial Test Sample", "📂 Upload a Raw Radiograph File"],
            index=0
        )
        
        selected_img_path = None
        
        if source_mode == "✨ Select Preloaded Industrial Test Sample":
            st.info("💡 Preloaded samples are taken directly from the project's holdout evaluation dataset.")
            sample_choice = st.selectbox(
                "Choose defect category to load:",
                [
                    "Crack (CR) - Critical rupture defect",
                    "Lack of Penetration (LP) - Serious bonding failure",
                    "Porosity (PO) - Trapped gas cavities",
                    "No Defect (ND) - Healthy weld structure"
                ]
            )
            
            # Map choice to code
            code_map = {
                "Crack (CR) - Critical rupture defect": "CR",
                "Lack of Penetration (LP) - Serious bonding failure": "LP",
                "Porosity (PO) - Trapped gas cavities": "PO",
                "No Defect (ND) - Healthy weld structure": "ND"
            }
            code = code_map[sample_choice]
            
            # Resolve image
            resolved_path = get_sample_image(code)
            if resolved_path:
                selected_img_path = resolved_path
                st.success(f"Successfully loaded validation sample: `{os.path.basename(resolved_path)}`")
            else:
                st.warning(f"Could not find test sample directory for class '{code}' in data/processed/test/.")
                
        else:
            uploaded_file = st.file_uploader(
                "Drag & drop or click to upload weld radiograph...",
                type=["png", "jpg", "jpeg"]
            )
            if uploaded_file is not None:
                # Save temporarily inside the scratch or workspace folder
                temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "temp"))
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                selected_img_path = temp_path
                st.success(f"Uploaded custom image file: `{uploaded_file.name}`")
        
        # Display Input Image
        if selected_img_path and os.path.exists(selected_img_path):
            st.markdown("##### 🔍 Input Radiograph Preview")
            img = Image.open(selected_img_path)
            st.image(img, use_column_width=True, caption=f"Source: {os.path.basename(selected_img_path)}")
            
            st.markdown("---")
            show_xai = st.checkbox("🧠 Show Explainable AI (XAI) Saliency Map", value=False, help="Enable to visualize the attention heatmap of the neural network on the weld joint.")
            
    with col_result:
        st.markdown("#### ⚡ AI Defect Diagnostic Engine")
        
        if selected_img_path is None:
            st.markdown(
                """
                <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 40px; text-align: center; color: #94a3b8;">
                    <span style="font-size: 3rem;">📥</span>
                    <h4 style="margin-top: 15px; color: #f8fafc; font-weight: 500;">Awaiting Input Image</h4>
                    <p style="font-size: 0.95rem; max-width: 320px; margin: 8px auto 0 auto; line-height: 1.5;">
                        Please select a preloaded sample or upload a raw file on the left side to trigger AI analysis.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            if predictor is None:
                st.error("Model predictor not initialized. Check if best.pt is trained and available.")
            else:
                with st.spinner("🔧 Analyzing weld structure... Running neural inference..."):
                    try:
                        pred = predictor.predict(selected_img_path)
                        
                        # Custom HTML card injection
                        st.markdown(
                            f"""
                            <div style="
                                border: 2px solid {pred['border_color']};
                                border-radius: 12px;
                                background-color: {pred['bg_color']};
                                padding: 24px;
                                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
                                margin-bottom: 25px;
                                transition: transform 0.3s;
                            ">
                                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 12px;">
                                    <span style="
                                        font-size: 0.85rem;
                                        font-weight: 700;
                                        background-color: {pred['color']};
                                        color: #0f172a;
                                        padding: 5px 12px;
                                        border-radius: 6px;
                                        text-transform: uppercase;
                                        letter-spacing: 0.05em;
                                    ">{pred['badge']}</span>
                                    <span style="font-size: 1rem; color: #94a3b8;">Confidence: <strong style="color: #f8fafc; font-size: 1.15rem;">{pred['confidence'] * 100:.2f}%</strong></span>
                                </div>
                                <h3 style="margin-top: 18px; font-weight: 700; color: #f8fafc; font-size: 1.6rem; margin-bottom: 8px;">{pred['class_name']}</h3>
                                <p style="font-size: 0.98rem; color: #cbd5e1; line-height: 1.6; margin-bottom: 20px;">{pred['description']}</p>
                                <div style="
                                    padding: 16px;
                                    background-color: rgba(15, 23, 42, 0.8);
                                    border-left: 5px solid {pred['color']};
                                    border-radius: 6px;
                                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);
                                ">
                                    <strong style="color: #f8fafc; font-size: 0.95rem; display: flex; align-items: center; gap: 6px;">👨‍🔧 Welding Inspection Action Protocol:</strong>
                                    <p style="margin: 8px 0 0 0; font-size: 0.92rem; color: #cbd5e1; line-height: 1.5;">{pred['recommendation']}</p>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Custom styled Probability bars
                        st.markdown("#### 📊 Confidence Spectrum Breakdown")
                        for class_name, probability in pred["all_probabilities"].items():
                            percentage = probability * 100
                            # Assign bar color matching class code style
                            bar_color = "#10B981"  # ND Green
                            if "Crack" in class_name:
                                bar_color = "#EF4444"
                            elif "Lack of Penetration" in class_name:
                                bar_color = "#F59E0B"
                            elif "Porosity" in class_name:
                                bar_color = "#F97316"
                                
                            st.markdown(
                                f"""
                                <div style="margin-bottom: 14px;">
                                    <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 4px; color: #cbd5e1;">
                                        <span>{class_name}</span>
                                        <strong style="color: #f8fafc;">{percentage:.2f}%</strong>
                                    </div>
                                    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 5px; height: 10px; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.5);">
                                        <div style="background-color: {bar_color}; width: {percentage}%; height: 100%; border-radius: 4px; transition: width 0.8s ease-out;"></div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                        # Explainable AI (XAI) Grad-CAM Heatmap
                        if show_xai:
                            st.markdown("---")
                            st.markdown("#### 🧠 Neural Diagnostic Saliency Map (Grad-CAM)")
                            st.markdown(
                                "This saliency heatmap highlights the specific local features and regions of interest "
                                "that the deep learning model focused on to make its diagnosis."
                            )
                            with st.spinner("🧠 Generating Grad-CAM diagnostic heatmap..."):
                                try:
                                    # Create temporary path for Grad-CAM output
                                    temp_gradcam_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "temp", "gradcam"))
                                    os.makedirs(temp_gradcam_dir, exist_ok=True)
                                    gradcam_path = os.path.join(temp_gradcam_dir, f"gradcam_{os.path.basename(selected_img_path)}")
                                    
                                    # Generate Grad-CAM image
                                    predictor.generate_gradcam(selected_img_path, output_path=gradcam_path)
                                    
                                    # Display Grad-CAM image
                                    gradcam_img = Image.open(gradcam_path)
                                    st.image(gradcam_img, use_column_width=True, caption=f"Grad-CAM Heatmap overlay (Class: {pred['class_name']})")
                                except Exception as e_cam:
                                    st.error(f"Failed to generate Grad-CAM visualization: {e_cam}")
                            
                    except Exception as ex:
                        st.error(f"Inference pipeline error: {ex}")

# ── PAGE 2: MODEL PERFORMANCE ANALYTICS ──────────────────────────
elif page == "📊 Model Performance Analytics":
    st.markdown("### 📊 Neural Network Performance & Training History")
    st.markdown("Examine the actual validation metrics, epoch loss traces, and diagnostic confusion matrices computed from the completed model training run.")
    
    # Custom HTML metrics grid
    st.markdown(
        """
        <div class="custom-card-grid">
            <div class="custom-metric-card">
                <div class="label">Architecture</div>
                <div class="value" style="color: #3b82f6;">YOLOv8-cls</div>
                <div class="subtext">Nano scale classification</div>
            </div>
            <div class="custom-metric-card">
                <div class="label">Training Epochs</div>
                <div class="value" style="color: #10b981;">50 Epochs</div>
                <div class="subtext">Completed training run</div>
            </div>
            <div class="custom-metric-card">
                <div class="label">Image Format</div>
                <div class="value" style="color: #f59e0b;">224 × 224</div>
                <div class="subtext">Auto-rescaled pixels</div>
            </div>
            <div class="custom-metric-card">
                <div class="label">Batch Size</div>
                <div class="value" style="color: #8b5cf6;">16 Images</div>
                <div class="subtext">Mini-batch size</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Resolve paths
    run_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "runs", "classify", "runs", "classify", "weld_v1"
    ))
    
    csv_path = os.path.join(run_dir, "results.csv")
    confusion_path = os.path.join(run_dir, "confusion_matrix_normalized.png")
    results_img_path = os.path.join(run_dir, "results.png")
    val_pred_path = os.path.join(run_dir, "val_batch0_pred.jpg")
    
    tab_curves, tab_matrices = st.tabs(["📉 Training History Curves", "🎯 Diagnostic Matrix & Sample Previews"])
    
    with tab_curves:
        col_curve_left, col_curve_right = st.columns(2)
        
        if os.path.exists(csv_path):
            try:
                # Load CSV logs
                df = pd.read_csv(csv_path)
                # Strip spaces from columns
                df.columns = df.columns.str.strip()
                
                with col_curve_left:
                    st.markdown("#### 📉 Dynamic Loss Traces")
                    st.markdown("Comparison between the model's training loss and its validation cross-entropy loss across all 50 epochs.")
                    
                    loss_df = df[['epoch', 'train/loss', 'val/loss']].copy()
                    loss_df.columns = ['Epoch', 'Training Loss', 'Validation Loss']
                    loss_df = loss_df.set_index('Epoch')
                    
                    st.line_chart(loss_df)
                    
                with col_curve_right:
                    st.markdown("#### 📈 Validation Accuracy Progress")
                    st.markdown("Top-1 accuracy metric computed on the validation split at the end of each epoch.")
                    
                    acc_df = df[['epoch', 'metrics/accuracy_top1']].copy()
                    acc_df.columns = ['Epoch', 'Top-1 Accuracy']
                    acc_df = acc_df.set_index('Epoch')
                    
                    st.line_chart(acc_df)
                    
            except Exception as e:
                st.error(f"Failed to load dynamic charts from CSV: {e}")
        else:
            st.warning(f"Training log file `results.csv` not found in `{run_dir}`. Dynamic line charts cannot be rendered.")
            
        st.markdown("---")
        st.markdown("#### 📈 Static YOLO Training Output (results.png)")
        if os.path.exists(results_img_path):
            st.image(Image.open(results_img_path), use_column_width=True, caption="YOLOv8 Automatic Training Metrics Panel")
        else:
            st.info("Static training results.png image not found in the run directory.")

    with tab_matrices:
        col_mat_left, col_mat_right = st.columns([1, 1])
        
        with col_mat_left:
            st.markdown("#### 🎯 Normalized Confusion Matrix")
            st.markdown("The confusion matrix displays the alignment between true labels (rows) and model predictions (columns). Values along the main diagonal represent the accuracy score per category.")
            
            if os.path.exists(confusion_path):
                st.image(Image.open(confusion_path), use_column_width=True, caption="Normalized Evaluation Matrix")
            else:
                st.warning("Normalized confusion matrix image not found.")
                
        with col_mat_right:
            st.markdown("#### 🔬 Neural Net Validation Batch Predictions")
            st.markdown("A composite snapshot of actual validation images showing true ground truth labels versus model confidence predictions, extracted from early validation logs.")
            
            if os.path.exists(val_pred_path):
                st.image(Image.open(val_pred_path), use_column_width=True, caption="Sample Predictions on Validation Batch")
            else:
                st.warning("Validation batch prediction preview image not found.")

# ── PAGE 3: METALLURGY & METHODOLOGY ──────────────────────────────
elif page == "📖 Metallurgy & Methodology":
    st.markdown("### 📖 Metallurgy & Defect Identification Standard")
    st.markdown("An inspection guide on the physical causes of welding defects, industrial quality thresholds, and radiographic detection.")
    
    st.markdown("---")
    
    col_met1, col_met2 = st.columns(2)
    
    with col_met1:
        st.markdown(
            """
            <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
                <h4 style="color: #ef4444; font-weight: 600; margin-top: 0;">🔴 Crack (CR)</h4>
                <p style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 0;">
                    <strong>Physical Cause:</strong> Occurs when localized stresses exceed the tensile strength of the weld metal. Cracks can be longitudinal or transverse. Hot cracking is caused by segregation of low-melting impurities (e.g. sulfur, phosphorus) during cooling, while cold cracking is often hydrogen-induced and occurs in the heat-affected zone (HAZ) hours after cooling.<br><br>
                    <strong>Radiographic Appearance:</strong> Appears as sharp, fine, dark, irregular linear indicators that are sometimes difficult to resolve if not oriented directly in line with the radiation beam.<br><br>
                    <strong>Engineering Severity:</strong> <strong>CRITICAL FAILURE.</strong> Cracks act as sharp stress concentration points and will propagate rapidly under mechanical stress or vibration, causing absolute failure.
                </p>
            </div>
            
            <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 20px;">
                <h4 style="color: #f59e0b; font-weight: 600; margin-top: 0;">🟡 Lack of Penetration (LP)</h4>
                <p style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 0;">
                    <strong>Physical Cause:</strong> Results when the weld metal does not extend completely through the full joint thickness or fails to bond completely at the root. Often caused by low heat input (voltage/amperage), excessive travel speed, incorrect electrode sizing, or poor joint preparation (e.g. root gap too narrow).<br><br>
                    <strong>Radiographic Appearance:</strong> Characterized by a highly distinct, straight dark line running down the precise center axis of the weld joint.<br><br>
                    <strong>Engineering Severity:</strong> <strong>SERIOUS.</strong> Reduces the structural load capacity of the joint significantly and invites crevice corrosion. Not acceptable for high-pressure boilers or chemical piping.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_met2:
        st.markdown(
            """
            <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
                <h4 style="color: #f97316; font-weight: 600; margin-top: 0;">🟠 Porosity (PO)</h4>
                <p style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 0;">
                    <strong>Physical Cause:</strong> Gas pockets trapped in the solidification zone. Typically formed by moisture on the welding electrode, contamination of the base metal (grease, rust, paint), or inadequate shielding gas flow in MIG/TIG processes, which permits atmospheric absorption.<br><br>
                    <strong>Radiographic Appearance:</strong> Appears as small, dark, circular spots that can be scattered randomly, clustered in specific locations, or aligned linearly along the weld track.<br><br>
                    <strong>Engineering Severity:</strong> <strong>MODERATE.</strong> Scattered pores are often acceptable in limited sizes, but clustered or linear porosity reduces the effective shear area of the weld, leading to joint rejection under ISO 5817 regulations.
                </p>
            </div>
            
            <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 20px;">
                <h4 style="color: #10b981; font-weight: 600; margin-top: 0;">🟢 No Defect (ND)</h4>
                <p style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 0;">
                    <strong>Physical Cause:</strong> An optimal weld joint executed with proper heat settings, shield gas protection, electrode quality, and welding technique.<br><br>
                    <strong>Radiographic Appearance:</strong> Smooth, uniform radiographic density across the weld bead, showing no linear anomalies, circular voids, or dark centerline indicators.<br><br>
                    <strong>Engineering Severity:</strong> <strong>STRUCTURAL PASS.</strong> Weld exhibits sound mechanical properties matching or exceeding the parent metal specifications. Safe for service.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("### 🏢 Industrial Inspection Standards Reference")
    st.markdown(
        """
        To ground the model predictions in industrial reality, our diagnostic actions correspond to:
        * **ISO 5817**: Joint quality levels for arc-welded joints in steel, nickel, titanium, and their alloys. (Levels B, C, and D).
        * **ASME Boiler & Pressure Vessel Code (BPVC) Section VIII / V**: Mandatory radiographic standards for pressurized system welds.
        * **AWS D1.1**: Structural Welding Code - Steel, governing structural bridges, buildings, and critical industrial scaffolding.
        """
    )

# ── FOOTER BAR ───────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer-bar">
        🔬 Weld Defect Radiographic Analyzer Dashboard • Designed for Quality Assurance and Non-Destructive Testing (NDT)
    </div>
    """,
    unsafe_allow_html=True
)
