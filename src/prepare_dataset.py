import os
import shutil

# ── CONFIG ──────────────────────────────────────────────
RAW_BASE   = r"C:\Users\nisha\Downloads\DB - Copy"
PROCESSED  = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

SPLIT_MAP  = {
    "training":   "train",
    "validation": "val",
    "testing":    "test"
}

CLASS_MAP  = {
    "Difetto1":  "CR",
    "Difetto2":  "LP",
    "Difetto4":  "PO",
    "NoDifetto": "ND"
}
# ────────────────────────────────────────────────────────

def prepare():
    for raw_split, yolo_split in SPLIT_MAP.items():
        for raw_class, yolo_class in CLASS_MAP.items():

            src = os.path.join(RAW_BASE, raw_split, raw_class)
            dst = os.path.join(PROCESSED, yolo_split, yolo_class)

            if not os.path.exists(src):
                print(f"[SKIP] Not found: {src}")
                continue

            os.makedirs(dst, exist_ok=True)
            files = os.listdir(src)

            for f in files:
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

            print(f"[OK] {raw_split}/{raw_class} -> {yolo_split}/{yolo_class} ({len(files)} images)")

if __name__ == "__main__":
    prepare()
    print("\nDataset ready at data/processed/") 