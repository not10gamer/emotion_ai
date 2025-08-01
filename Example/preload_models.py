import logging
import numpy as np
from deepface import DeepFace

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preload():
    """
    Preloads the DeepFace models to avoid race conditions when running with multiple
    gunicorn workers.
    """
    try:
        logger.info("Starting DeepFace model preloading...")
        # Use a dummy frame to trigger the model download and initialization.
        dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.analyze(dummy_frame, actions=['emotion'], enforce_detection=False, silent=True)
        logger.info("DeepFace models preloaded successfully.")
    except Exception as e:
        logger.error(f"Could not preload DeepFace models: {e}", exc_info=True)
        # Exit with a non-zero code to indicate failure, which can be useful in CI/CD pipelines.
        exit(1)

if __name__ == '__main__':
    preload()
