import requests

CLOUD_URL = "http://3.8.74.239:8000"


def get_cloud_health():
    response = requests.get(
        f"{CLOUD_URL}/health",
        timeout=10
    )
    response.raise_for_status()
    return response.json()


def get_cloud_telemetry():
    response = requests.get(
        f"{CLOUD_URL}/telemetry",
        timeout=10
    )
    response.raise_for_status()
    return response.json()


def run_cloud_inference(model_size_mb=6.2, batch_size=1):
    response = requests.post(
        f"{CLOUD_URL}/inference",
        json={
            "workload_type": "compute",
            "model_size_mb": float(model_size_mb),
            "batch_size": int(batch_size),
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()



def run_cloud_vision(
    image_bytes,
    filename="camera-frame.jpg",
    content_type="image/jpeg",
    confidence=0.25,
):
    response = requests.post(
        f"{CLOUD_URL}/vision",
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


def run_cloud_florence(
    image_bytes,
    filename="camera-frame.jpg",
    content_type="image/jpeg",
    task_prompt="<MORE_DETAILED_CAPTION>",
):
    response = requests.post(
        f"{CLOUD_URL}/florence",
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
    print("HEALTH:")
    print(get_cloud_health())

    print("\nTELEMETRY:")
    print(get_cloud_telemetry())

    print("\nINFERENCE:")
    print(run_cloud_inference())

