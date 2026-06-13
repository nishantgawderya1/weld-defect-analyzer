import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

def generate_ablation_study(v1_csv_path: str, v2_csv_path: str, output_path: str):
    """
    Plots validation accuracy and training/validation loss curves of 
    both baseline and preprocessed models side-by-side.
    """
    if not os.path.exists(v1_csv_path):
        print(f"[ERROR] Baseline results.csv not found at: {v1_csv_path}")
        return False
        
    if not os.path.exists(v2_csv_path):
        print(f"[ERROR] Preprocessed results.csv not found at: {v2_csv_path}")
        return False

    print(f"[INFO] Loading training logs...")
    # Read training result CSVs
    v1_df = pd.read_csv(v1_csv_path)
    v2_df = pd.read_csv(v2_csv_path)

    # Strip column names of whitespace just in case
    v1_df.columns = v1_df.columns.str.strip()
    v2_df.columns = v2_df.columns.str.strip()

    # Verify column metrics exist
    acc_col = "metrics/accuracy_top1"
    loss_col = "val/loss"

    if acc_col not in v1_df.columns or acc_col not in v2_df.columns:
        print(f"[ERROR] Required column '{acc_col}' not found in CSV. Found: {list(v1_df.columns)}")
        return False

    # Establish plotting grid
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Validation Accuracy Comparison
    axes[0].plot(v1_df["epoch"], v1_df[acc_col], label="Baseline Model (No Preprocessing)", color="#ef4444", linewidth=2)
    axes[0].plot(v2_df["epoch"], v2_df[acc_col], label="Preprocessed Model (CLAHE + Denoise)", color="#10b981", linewidth=2)
    axes[0].set_title("Validation Accuracy (Top-1)", fontsize=13, fontweight='bold', pad=15)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=10, loc="lower right")

    # Plot Validation Loss Comparison
    axes[1].plot(v1_df["epoch"], v1_df[loss_col], label="Baseline Model Loss", color="#f59e0b", linewidth=2)
    axes[1].plot(v2_df["epoch"], v2_df[loss_col], label="Preprocessed Model Loss", color="#3b82f6", linewidth=2)
    axes[1].set_title("Validation Cross-Entropy Loss", fontsize=13, fontweight='bold', pad=15)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=10, loc="upper right")

    # Overall plot layout tuning
    plt.suptitle("Weld AI (ADR) Preprocessing Ablation Study", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Ensure output folder exists and save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"[SUCCESS] Ablation study comparison plot saved to: {output_path}")
    return True


if __name__ == "__main__":
    # Resolve absolute paths from workspace root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Try different potential CSV log locations for weld_v1 (baseline)
    csv_locations = [
        os.path.join(base_dir, "runs", "classify", "runs", "classify", "weld_v1", "results.csv"),
        os.path.join(base_dir, "runs", "classify", "weld_v1", "results.csv")
    ]
    
    v1_csv = None
    for loc in csv_locations:
        if os.path.exists(loc):
            v1_csv = loc
            break

    # Try different potential CSV log locations for weld_v2_preprocessed
    v2_csv_locations = [
        os.path.join(base_dir, "runs", "classify", "runs", "classify", "weld_v2_preprocessed", "results.csv"),
        os.path.join(base_dir, "runs", "classify", "weld_v2_preprocessed", "results.csv")
    ]
    
    v2_csv = None
    for loc in v2_csv_locations:
        if os.path.exists(loc):
            v2_csv = loc
            break
    
    # Output path
    out_plot = os.path.join(base_dir, "outputs", "ablation_study.png")

    print(f"Comparing training curves:\n - Baseline log: {v1_csv}\n - Preprocessed log: {v2_csv}\n - Export plot: {out_plot}")
    
    if v1_csv and v2_csv and os.path.exists(v2_csv):
        generate_ablation_study(v1_csv_path=v1_csv, v2_csv_path=v2_csv, output_path=out_plot)
    else:
        print("[WARNING] Could not execute ablation study comparison yet. Ensure both models have completed training.")
