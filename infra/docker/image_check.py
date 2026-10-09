"""Smoke test for the backend image: the ml stack imports and runs on the CPU, with the system libraries this image has.

The Dockerfile runs it at build time, so a CUDA build of PyTorch or a missing shared library fails the build (and CI,
which builds the image) instead of showing up as "provider unavailable" at runtime. It can be re-run in a container:

    docker compose -f infra/docker-compose.yml exec backend python /app/image_check.py

Loads no model weights (those are mounted from data/models), so it needs well under 1 GB.
"""

from __future__ import annotations

import io
import sys
import sysconfig
from pathlib import Path


def torch_is_cpu_only() -> str:
    import torch

    assert torch.version.cuda is None and not torch.cuda.is_available(), f"CUDA build of PyTorch: {torch.__version__}"
    return f"torch {torch.__version__}"


def ml_libraries_import() -> str:
    import docling  # noqa: F401
    import faster_whisper  # noqa: F401
    import FlagEmbedding  # noqa: F401
    import kokoro  # noqa: F401
    import onnxruntime
    import rapidocr  # noqa: F401
    import sentence_transformers  # noqa: F401

    return f"onnxruntime {onnxruntime.__version__}"


def audio_and_vision_libraries_run() -> str:
    import cv2
    import numpy as np
    import soundfile

    buffer = io.BytesIO()
    soundfile.write(buffer, np.zeros(1600, dtype="float32"), 16000, format="WAV")
    ok, _ = cv2.imencode(".png", np.zeros((8, 8, 3), dtype="uint8"))
    assert ok
    return f"libsndfile {soundfile.__libsndfile_version__}, opencv {cv2.__version__}"


def kokoro_grapheme_to_phoneme() -> str:
    # Hindi goes through espeak-ng (the espeakng-loader wheel bundles it, no system package), English through spaCy.
    from misaki import en, espeak

    hindi, _ = espeak.EspeakG2P(language="hi")("नमस्ते")
    english, _ = en.G2P(trf=False, british=False, fallback=None)("Hello")
    assert hindi and english
    return f"hi {hindi!r}, en {english!r}"


def silero_vad_runs() -> str:
    # The server-side VAD: ONNX Runtime on the model bundled in the silero-vad wheel.
    import onnxruntime

    model = Path(sysconfig.get_paths()["purelib"]) / "silero_vad" / "data" / "silero_vad.onnx"
    onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    return model.name


def main() -> int:
    failed = 0
    for check in (
        torch_is_cpu_only,
        ml_libraries_import,
        audio_and_vision_libraries_run,
        kokoro_grapheme_to_phoneme,
        silero_vad_runs,
    ):
        try:
            print(f"ok    {check.__name__}: {check()}")
        except Exception as e:  # report every failing check, not just the first
            failed += 1
            print(f"FAIL  {check.__name__}: {type(e).__name__}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
