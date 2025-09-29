import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging
import torch
import torchvision
import json
import cv2
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Logging setup
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Load class mappings
# --------------------------------------------------
with open("class_mapping.json") as f:
    mappings = json.load(f)
class_mapping = {item["model_idx"]: item["class_name"] for item in mappings}

# --------------------------------------------------
# Model Wrapper
# --------------------------------------------------
class MyModel:
    def __init__(self, model_path: str = "model.pt", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading TorchScript model from {model_path} on {self.device}...")
        self.model = torch.jit.load(model_path).to(self.device)
        logger.info("TorchScript model loaded successfully.")

    def predict_and_draw(self, image: Image.Image, save_path: Path, target_class_idx: int = 0):
        """
        Run inference, apply NMS, map class IDs → names,
        draw bounding boxes only for the target class if specified,
        and save it.
        """
        np_img = np.array(image)
        tensor_img = torch.from_numpy(np_img).to(self.device)

        with torch.no_grad():
            # Convert to channels-first and float
            tensor_img = tensor_img.permute(2, 0, 1).float()
            outputs = self.model(tensor_img)

        # Apply NMS
        keep = torchvision.ops.nms(outputs["pred_boxes"], outputs["scores"], 0.5)
        boxes = outputs["pred_boxes"][keep]
        classes = outputs["pred_classes"][keep]
        scores = outputs["scores"][keep]

        predictions = []
        for bbox, cls, score in zip(boxes, classes, scores):
            class_idx = int(cls.item())

            # 🔥 Only keep target class if specified
            if target_class_idx is not None and class_idx != target_class_idx:
                continue

            x1, y1, x2, y2 = map(int, bbox.tolist())
            class_name = class_mapping.get(class_idx, str(class_idx))
            accuracy = float(score.item())

            predictions.append({
                "class_id": class_idx,
                "class_name": class_name,
                "score": accuracy,
                "bbox": [x1, y1, x2, y2]
            })

            # Draw bounding box + label
            cv2.rectangle(np_img, (x1, y1), (x2, y2), (255, 0, 0), 4)
            cv2.putText(
                np_img,
                f"{class_name} {accuracy:.2f}",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 0, 0),
                2
            )

        # Save the labeled image back
        cv2.imwrite(str(save_path), cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))

        return predictions

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(title="Batch Labeling Model Server")

# Load model once
model = MyModel("model.pt")

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Ensure save dir
        save_dir = Path("loadedimages")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / file.filename

        # Run prediction + draw bounding boxes
        preds = model.predict_and_draw(image, save_path)

        logger.info(f"Image saved with labels: {save_path}")
        logger.info(f"Predictions: {preds}")

        return JSONResponse({"predictions": preds, "saved_image": str(save_path)})

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# --------------------------------------------------
# Entry Point
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
