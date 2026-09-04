from pathlib import Path
from time import perf_counter
from typing import Any

import psutil
import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration


class FlorenceRunner:
    """Runs Florence-2 locally for detailed image descriptions."""

    def __init__(
        self,
        model_name: str = "florence-community/Florence-2-base",
    ) -> None:
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        self.processor: AutoProcessor | None = None
        self.model: Florence2ForConditionalGeneration | None = None

    def _load_model(self) -> None:
        """Load the processor and model once, then reuse them."""

        if self.model is not None and self.processor is not None:
            return

        print(
            f"Loading Florence-2 model: {self.model_name} "
            f"on device: {self.device}"
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
        )

        self.model = (
            Florence2ForConditionalGeneration
            .from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
            )
            .to(self.device)
        )

        self.model.eval()

        print("Florence-2 model loaded successfully.")

    def analyse(
        self,
        image_path: str,
        task_prompt: str = "<MORE_DETAILED_CAPTION>",
    ) -> dict[str, Any]:
        """
        Analyse an image and return a detailed visual description.

        Florence-2 supports task prompts such as:
        - <CAPTION>
        - <DETAILED_CAPTION>
        - <MORE_DETAILED_CAPTION>
        - <OD>
        - <DENSE_REGION_CAPTION>
        """

        self._load_model()

        if self.processor is None or self.model is None:
            raise RuntimeError(
                "Florence-2 model was not loaded correctly."
            )

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path.resolve()}"
            )

        if not path.is_file():
            raise ValueError(
                f"Image path is not a file: {path.resolve()}"
            )

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise ValueError(
                f"Unable to open image: {path.resolve()}"
            ) from exc

        # -----------------------------------------------------
        # PREPARE FLORENCE INPUT
        # -----------------------------------------------------

        inputs = self.processor(
            text=task_prompt,
            images=image,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)

        pixel_values = inputs["pixel_values"].to(
            device=self.device,
            dtype=self.dtype,
        )

        attention_mask = inputs.get("attention_mask")

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # -----------------------------------------------------
        # REAL RESOURCE MEASUREMENT - BEFORE INFERENCE
        # -----------------------------------------------------

        process = psutil.Process()

        memory_before_mb = (
            process.memory_info().rss / (1024 * 1024)
        )

        cpu_times_before = process.cpu_times()

        if self.device == "cuda":
            torch.cuda.synchronize()
            gpu_memory_before_mb = (
                torch.cuda.memory_allocated() / (1024 * 1024)
            )
        else:
            gpu_memory_before_mb = 0.0

        wall_start = perf_counter()

        # -----------------------------------------------------
        # REAL FLORENCE-2 INFERENCE
        # -----------------------------------------------------

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
                early_stopping=True,
            )

        # CUDA operations are asynchronous.
        # Synchronise before stopping the timer.
        if self.device == "cuda":
            torch.cuda.synchronize()

        wall_end = perf_counter()

        # -----------------------------------------------------
        # REAL RESOURCE MEASUREMENT - AFTER INFERENCE
        # -----------------------------------------------------

        cpu_times_after = process.cpu_times()

        latency_ms = (wall_end - wall_start) * 1000.0

        cpu_time_used = (
            (cpu_times_after.user - cpu_times_before.user)
            + (
                cpu_times_after.system
                - cpu_times_before.system
            )
        )

        wall_seconds = max(
            wall_end - wall_start,
            1e-9,
        )

        # Normalised process CPU utilisation across logical CPUs.
        cpu_count = psutil.cpu_count(logical=True) or 1

        cpu_percent = (
            (cpu_time_used / wall_seconds) * 100.0
        ) / cpu_count

        cpu_percent = min(
            max(cpu_percent, 0.0),
            100.0,
        )

        memory_after_mb = (
            process.memory_info().rss / (1024 * 1024)
        )

        if self.device == "cuda":
            gpu_memory_after_mb = (
                torch.cuda.memory_allocated() / (1024 * 1024)
            )
        else:
            gpu_memory_after_mb = 0.0

        # -----------------------------------------------------
        # PROCESS FLORENCE OUTPUT
        # -----------------------------------------------------

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]

        parsed_result = (
            self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(
                    image.width,
                    image.height,
                ),
            )
        )

        description = parsed_result.get(
            task_prompt,
            parsed_result,
        )

        # One visual assessment completed in latency_ms.
        throughput_fps = (
            1000.0 / max(latency_ms, 0.001)
        )

        # -----------------------------------------------------
        # RESULT
        # -----------------------------------------------------

        return {
            "success": True,

            "model_name": self.model_name,
            "device": self.device,
            "task": task_prompt,

            "image_path": str(path),
            "image_width": image.width,
            "image_height": image.height,

            "description": description,

            # Real measured inference performance
            "latency_ms": round(latency_ms, 2),
            "cpu_percent": round(cpu_percent, 2),

            "memory_before_mb": round(
                memory_before_mb,
                2,
            ),
            "memory_after_mb": round(
                memory_after_mb,
                2,
            ),
            "memory_delta_mb": round(
                memory_after_mb - memory_before_mb,
                2,
            ),

            # CUDA memory, when available
            "gpu_memory_before_mb": round(
                gpu_memory_before_mb,
                2,
            ),
            "gpu_memory_after_mb": round(
                gpu_memory_after_mb,
                2,
            ),
            "gpu_memory_delta_mb": round(
                gpu_memory_after_mb
                - gpu_memory_before_mb,
                2,
            ),

            # Derived throughput
            "fps_estimate": round(
                throughput_fps,
                4,
            ),

            "detail": (
                "Florence-2 local visual assessment "
                f"on {self.device}"
            ),
        }


florence_runner = FlorenceRunner()