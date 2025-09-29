<<<<<<< HEAD
# batchLabellingPipelineImages
=======
# 📦 Batch Labeling Pipeline  

*A Task Submission for CloudFactory*  

This project implements a **production-oriented batch image labeling pipeline**.  
It is designed to take a set of input images, preprocess them, run inference using a hosted ML model, and output predictions in **COCO format**, with complete logging and teardown steps.  

---

## 🚀 Task Objectives  

Given:  
- A Machine Learning model  
- A set of images to label  

We must build a pipeline that:  
1. Deploys infrastructure to host the model for inference  
2. Processes raw images with augmentation (image preprocessing)  
3. Produces output predictions in **COCO format**  
4. Tears down infrastructure after task completion  

---

## 🏗️ Architecture  

### Diagram  

```text
 +------------------+        +--------------------+        +--------------------+
 |                  |        |                    |        |                    |
 |  Input Images    | -----> |  Preprocessing     | -----> |  Model Server      |
 |(data/sampleimages)        |  (pipeline/preprocess.py) |  |  (FastAPI + Model) |
 |                  |        |                    |        |                    |
 +------------------+        +--------------------+        +--------------------+
                                                                |
                                                                |
                                                                v
                                                     +--------------------+
                                                     |  COCO Export        |
                                                     |  (pipeline/coco_utils.py) |
                                                     +--------------------+
                                                                |
                                                                v
                                                     +--------------------+
                                                     |  Output            |
                                                     |  predictions.json  |
                                                     |  run.log           |
                                                     +--------------------+
````

### Components

* **Input Images** — Folder with `.jpg`, `.jpeg`, `.png` images.
* **Preprocessing** — Image augmentation & preprocessing with OpenCV. Supports resizing, RGB conversion, and configurable transforms.
* **Model Server** — FastAPI app serving the ML model for inference. Dockerized for portability.
* **COCO Export** — Converts predictions into COCO-style JSON (`images`, `annotations`, `categories`).
* **Output** — Stores predictions (`predictions.json`) and logs (`run.log`).

---

## ✨ Features

* 🔧 **Dockerized model server** for easy deployment
* 🖼️ **OpenCV preprocessing** with batch and augmentation support
* 📑 **COCO format output** ready for ML pipelines
* 📜 **Detailed logging** (`output/run.log`)
* 🧪 **Unit + Integration tests** (pytest)
* ⚙️ **GitHub Actions CI** for linting + tests
* 🔄 **Extensible** — swap in real ML models (`.pt`, `.onnx`, `.h5`)

---

## ⚡ Advantages of this Architecture

* **Modular** — Each step (preprocessing, server, batch runner, export) is independent
* **Scalable** — Handles large datasets with configurable `--batch_size`
* **Reproducible** — Docker ensures consistent runtime
* **Standardized** — COCO format ensures downstream compatibility
* **Extensible** — Replace dummy model with PyTorch, TensorFlow, or ONNX models
* **Robust** — Logging + tests ensure reliability in production workflows

---

## 🛠️ Requirements

* **Python**: 3.10+
* **Docker & Docker Compose** (for containerized execution)
* **Dependencies**:

  ```bash
  pip install -r requirements.txt
  pip install -r model_server/requirements.txt
  ```

---


### Run with Docker 

1. **Build & start server**

   ```bash
   docker-compose up --build -d
   ```

2. **Run batch pipeline**

   ```bash
   python -m pipeline.run_batch \
       --input_dir data/sample_images \
       --output_file output/predictions.json \
       --batch_size 4 \
       --server_url http://localhost:8000/predict/
   ```

3. **Check outputs**

   ```bash
   less output/predictions.json   # COCO results
   less output/run.log            # Logs
   ```

4. **Tear down**

   ```bash
   docker-compose down
   ```

---

### Run Locally (Without Docker)

1. **Start server**

   ```bash
   python -m model_server.main
   ```

2. **Run pipeline** (new terminal)

   ```bash
   python -m pipeline.run_batch --input_dir data/sample_images --output_file output/predictions.json --batch_size 1 --server_url http://localhost:8000/predict/
   ```

---


---

## 🧪 Running Tests

```bash
pytest -q
```

Covers:

* ✅ Preprocessing
* ✅ COCO export
* ✅ Model server + pipeline integration

---

## 📂 Output Files

* **`output/predictions.json`** — COCO predictions (images + annotations).
* **`output/run.log`** — Detailed logs with timestamps.






>>>>>>> 992bd08 (task completed)
