import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def save_predictions_coco(predictions, output_path: str):
    images = []
    annotations = []
    ann_id = 1
    for i, p in enumerate(predictions, start=1):
        images.append({
            "id": i,
            "file_name": p["file_name"],
            "width": p.get("width", -1),
            "height": p.get("height", -1)
        })
        for ann in p.get("annotations", []):
            bbox = ann.get("bbox", [0,0,0,0])
            annotations.append({
                "id": ann_id,
                "image_id": i,
                "category_id": ann.get("category_id", 1),
                "bbox": bbox,
                "score": ann.get("score", 0.0)
            })
            ann_id += 1
    coco = {"images": images, "annotations": annotations, "categories": [{"id":1,"name":"object"}]}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)
    logger.info(f"Saved COCO predictions to {output_path}")
