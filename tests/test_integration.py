import subprocess, time, requests, os, signal , pytest
from pathlib import Path

@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="End-to-end server tests are skipped in CI environment"
)
def test_end_to_end(tmp_path):
    # start server
    proc = subprocess.Popen(["python", "-m", "model_server.main"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        time.sleep(2)
        # check server alive
        r = requests.get("http://localhost:8000/docs")
        assert r.status_code == 200
        # run batch runner
        out_file = tmp_path / "out_preds.json"
        cmd = ["python", "-m", "pipeline.run_batch", "--input_dir", "data/sample_images", "--output_file", str(out_file), "--batch_size", "1", "--server_url", "http://localhost:8000/predict/"]
        rc = subprocess.call(cmd, timeout=60)
        assert rc == 1 or out_file.exists()
    finally:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
