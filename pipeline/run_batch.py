"""
Run batch inference against the model server and export COCO-format results.

Usage:
    python -m pipeline.run_batch \
        --input_dir data/sample_images \
        --output_file output/predictions.json \
        --batch_size 8 \
        --server_url http://localhost:8000/predict/
"""

import argparse
import logging
from pathlib import Path
import requests
from pipeline.preprocess import preprocess_image
from pipeline.coco_utils import save_predictions_coco
import cv2

def setup_logger(log_file: Path) -> logging.Logger:
    """Configure a logger that writes to both file and console."""
    logger = logging.getLogger("batch_runner")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(n):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk

def run(input_dir: Path, output_file: Path, server_url: str, batch_size: int, tmp_dir: Path):
    """Main batch processing function."""
    # Setup logging
    log_file = Path("output/run.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file)

    logger.info(f"Starting batch run on {input_dir} -> {output_file}")

    # Gather image files
    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    logger.info(f"Found {len(images)} images")
    if not images:
        logger.warning("No images found. Exiting...")
        return

    tmp_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for batch in chunked(images, batch_size):
        files = []
        proc_paths = []

        # Preprocess images
        for img_path in batch:
            try:
                img_path = Path(img_path)
                proc_path = tmp_dir / img_path.name

                # Preprocess and resize image
                img = preprocess_image(str(img_path), size=(1000, 1000), keep_aspect=True)

                # Save preprocessed image to temporary folder
                cv2.imwrite(str(proc_path), img)
                proc_paths.append(proc_path)

                # Prepare file tuple for POST
                files.append(('file', (proc_path.name, open(proc_path, 'rb'), 'image/jpeg')))

            except Exception as e:
                logger.error(f"Failed to preprocess {img_path}: {e}")

        # Send batch to model server
        if files:
            try:
                resp = requests.post(server_url, files=files, timeout=35)
                resp.raise_for_status()
                preds = resp.json().get('predictions', [])

                for img_path, pred in zip(batch, preds):
                    results.append({
                        'file_name': Path(img_path).name,
                        'annotations': pred if isinstance(pred, list) else [pred]
                    })
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for batch: {e}")
            finally:
                # Safely close all file objects
                for _, ftuple in files:
                    try:
                        ftuple[1].close()
                    except Exception:
                        pass

    # Save results in COCO format
    save_predictions_coco(results, str(output_file))
    logger.info(f"Batch run completed successfully. Output written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference and export COCO-format predictions")
    parser.add_argument('--input_dir', required=True, type=Path, help="Directory containing input images")
    parser.add_argument('--output_file', required=True, type=Path, help="File path to save predictions (JSON)")
    parser.add_argument('--server_url', default='http://localhost:8000/predict/', help="Model server prediction URL")
    parser.add_argument('--batch_size', default=8, type=int, help="Number of images to send in each batch")
    parser.add_argument('--tmp_dir', default=Path('tmp_proc'), type=Path, help="Temporary folder for preprocessed images")
    args = parser.parse_args()

    run(args.input_dir, args.output_file, args.server_url, args.batch_size, args.tmp_dir)
