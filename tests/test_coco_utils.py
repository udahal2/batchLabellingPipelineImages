from pipeline.coco_utils import save_predictions_coco
import json

def test_save_predictions_coco(tmp_path):
    preds = [{
        "file_name": "test.jpg",
        "annotations": [{"bbox": [1,2,3,4], "category_id": 1, "score": 0.9}]
    }]
    out_file = tmp_path / "preds.json"
    save_predictions_coco(preds, str(out_file))
    data = json.loads(out_file.read_text())
    assert "images" in data and "annotations" in data
