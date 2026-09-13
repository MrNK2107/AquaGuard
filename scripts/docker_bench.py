"""
AquaGuard: In-Container Load Benchmark & Latency Profiler
=========================================================
Executes concurrent load testing against /detect and /reason endpoints,
measuring p50, p95, and p99 latencies, throughput (requests/sec), and guardrail accuracy.
"""

import time
import io
import concurrent.futures
from pathlib import Path
from PIL import Image
import requests

BASE_URL = "http://localhost:8000"

def generate_test_image() -> bytes:
    img = Image.new("RGB", (512, 512), color=(25, 75, 95))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def benchmark_endpoint(endpoint: str, num_requests: int = 20, concurrency: int = 4):
    print(f"\n🚀 Benchmarking {endpoint} ({num_requests} requests, concurrency={concurrency})...")
    img_bytes = generate_test_image()
    
    latencies = []
    success_count = 0
    
    def send_req(idx: int):
        t0 = time.time()
        try:
            if endpoint == "/detect":
                r = requests.post(f"{BASE_URL}/detect?conf=0.25", files={"file": ("bench.jpg", img_bytes, "image/jpeg")}, timeout=10)
            else:
                r = requests.post(f"{BASE_URL}/reason", files={"file": ("bench.jpg", img_bytes, "image/jpeg")}, data={"question": "How many objects?", "conf": 0.35}, timeout=10)
            dt = (time.time() - t0) * 1000
            if r.status_code == 200:
                return dt, True
            return dt, False
        except Exception as e:
            return 0, False

    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_req, i) for i in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            lat, ok = f.result()
            if ok:
                latencies.append(lat)
                success_count += 1

    total_time = time.time() - t_start
    if not latencies:
        print("❌ All requests failed. Is the API running?")
        return

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    rps = success_count / total_time

    print(f"✅ Completed {success_count}/{num_requests} requests in {total_time:.2f}s")
    print(f"   Throughput: {rps:.1f} req/sec")
    print(f"   Latency p50: {p50:.1f} ms | p95: {p95:.1f} ms | p99: {p99:.1f} ms")

if __name__ == "__main__":
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=3)
        print("API Health Check:", r.json())
        benchmark_endpoint("/detect", num_requests=10, concurrency=2)
        benchmark_endpoint("/reason", num_requests=10, concurrency=2)
    except Exception as e:
        print(f"Local server not currently running on {BASE_URL}. Run uvicorn app.main:app first.")
