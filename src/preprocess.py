import cv2
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    CLAHE: Enhances local contrast in X-ray images.
    clip_limit: controls contrast enhancement (2.0 is standard for X-rays)
    tile_size: grid size for local histogram (8x8 is standard)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def apply_denoising(image: np.ndarray, h: int = 10) -> np.ndarray:
    """
    Non-local means denoising: removes X-ray film grain.
    h: filter strength (10 is standard, higher = more smoothing)
    """
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)


def preprocess_image(image_path: str) -> np.ndarray:
    """Full preprocessing pipeline for a single X-ray image."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")
    image = apply_clahe(image)
    image = apply_denoising(image)
    return image


def process_single_file(img_path_str: str, relative_str: str, output_dir_str: str) -> bool:
    """Processes a single image file and writes the output. Run in worker processes."""
    try:
        out_path = Path(output_dir_str) / relative_str
        out_path.parent.mkdir(parents=True, exist_ok=True)
        processed = preprocess_image(img_path_str)
        cv2.imwrite(str(out_path), processed)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to preprocess {img_path_str}: {e}")
        return False


def preprocess_dataset(input_dir: str, output_dir: str):
    """
    Applies preprocessing to entire dataset in parallel.
    Maintains same folder structure as input.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    image_files = list(input_path.rglob("*.png")) + list(input_path.rglob("*.jpg")) + list(input_path.rglob("*.jpeg"))
    num_files = len(image_files)
    print(f"[INFO] Found {num_files} images to preprocess")

    success_count = 0
    # Use max_workers=None to default to number of processors
    print("[INFO] Initializing multi-core preprocessing pool...")
    with ProcessPoolExecutor() as executor:
        futures = []
        for img_path in image_files:
            relative = img_path.relative_to(input_path)
            futures.append(
                executor.submit(
                    process_single_file,
                    str(img_path),
                    str(relative),
                    str(output_path)
                )
            )

        for idx, future in enumerate(as_completed(futures)):
            if future.result():
                success_count += 1
            if (idx + 1) % 2000 == 0 or (idx + 1) == num_files:
                print(f"[PROGRESS] Preprocessed {idx + 1}/{num_files} images... (Success: {success_count})")

    print(f"[DONE] Preprocessed {success_count}/{num_files} images successfully. Saved to: {output_dir}")


if __name__ == "__main__":
    # Resolve absolute paths from workspace root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_directory = os.path.join(base_dir, "data", "processed")
    output_directory = os.path.join(base_dir, "data", "preprocessed")

    print(f"Starting dataset preprocessing from {input_directory} to {output_directory}...")
    preprocess_dataset(input_dir=input_directory, output_dir=output_directory)
