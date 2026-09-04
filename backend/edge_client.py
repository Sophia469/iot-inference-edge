import requests

# Edge Compute Node API
EDGE_BASE_URL = "http://127.0.0.1:8000"


def get_edge_telemetry():
    response = requests.get(
        f"{EDGE_BASE_URL}/telemetry",
        timeout=3
    )
    response.raise_for_status()
    return response.json()


def get_edge_health():
    response = requests.get(
        f"{EDGE_BASE_URL}/health",
        timeout=3
    )
    response.raise_for_status()
    return response.json()


def run_edge_inference(model_size_mb=6.2, batch_size=1):
    response = requests.post(
        f"{EDGE_BASE_URL}/inference",
        json={
            "workload_type": "compute",
            "model_size_mb": float(model_size_mb),
            "batch_size": int(batch_size),
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()



def run_edge_vision(
    image_bytes,
    filename="camera-frame.jpg",
    content_type="image/jpeg",
    confidence=0.25,
):
    response = requests.post(
        f"{EDGE_BASE_URL}/vision",
        params={
            "confidence": float(confidence),
        },
        files={
            "file": (
                filename,
                image_bytes,
                content_type,
            )
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def run_edge_florence(
    image_bytes,
    filename="camera-frame.jpg",
    content_type="image/jpeg",
    task_prompt="<MORE_DETAILED_CAPTION>",
):
    response = requests.post(
        f"{EDGE_BASE_URL}/florence",
        params={
            "task_prompt": task_prompt,
        },
        files={
            "file": (
                filename,
                image_bytes,
                content_type,
            )
        },
        timeout=360,
    )
    response.raise_for_status()
    return response.json()
if __name__ == "__main__":
    print("Edge telemetry:")
    print(get_edge_telemetry())

    print("\nEdge health:")
    print(get_edge_health())

    print("\nEdge inference:")
    print(run_edge_inference())

