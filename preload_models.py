import logging
import numpy as np
import time
import os
from deepface import DeepFace
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def optimize_tensorflow():
    """Optimize TensorFlow settings for production"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Configured {len(gpus)} GPU(s) with memory growth")
        else:
            logger.info("No GPUs found, using CPU")

        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.threading.set_intra_op_parallelism_threads(0)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')

        logger.info("TensorFlow optimizations applied successfully")
        return True
    except Exception as e:
        logger.error(f"Error optimizing TensorFlow: {e}")
        return False


def preload_deepface_models():
    """
    Preloads all DeepFace models to avoid cold start delays.
    """
    try:
        logger.info("Starting DeepFace model preloading...")
        start_time = time.time()

        dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)

        # UPDATED: Preload using the more balanced mtcnn backend
        logger.info("Preloading emotion detection model with mtcnn backend...")
        result = DeepFace.analyze(
            dummy_frame,
            actions=['emotion'],
            enforce_detection=False,
            silent=True,
            detector_backend='mtcnn'
        )
        logger.info(f"Emotion model loaded. Sample result: {result[0]['dominant_emotion'] if result else 'None'}")

        loading_time = time.time() - start_time
        logger.info(f"DeepFace models preloaded successfully in {loading_time:.2f} seconds")
        return True

    except Exception as e:
        logger.error(f"Could not preload DeepFace models: {e}", exc_info=True)
        return False


def verify_model_functionality():
    """Verify that preloaded models work correctly"""
    try:
        logger.info("Verifying model functionality...")
        test_frame = np.ones((224, 224, 3), dtype=np.uint8) * 128

        # UPDATED: Verify using the more balanced mtcnn backend
        result = DeepFace.analyze(
            test_frame,
            actions=['emotion'],
            enforce_detection=False,
            silent=True,
            detector_backend='mtcnn'
        )

        if result and len(result) > 0:
            dominant_emotion = result[0].get('dominant_emotion', 'unknown')
            logger.info(f"Model verification successful. Detected emotion: {dominant_emotion}")
            return True
        else:
            logger.warning("Model verification returned empty result")
            return False

    except Exception as e:
        logger.error(f"Model verification failed: {e}", exc_info=True)
        return False


def check_system_resources():
    """Check available system resources"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        logger.info(
            f"System Resources: CPU cores: {cpu_count}, Available memory: {memory.available / (1024 ** 3):.2f} GB")
    except ImportError:
        logger.info("psutil not available, skipping resource check")
    except Exception as e:
        logger.error(f"Error checking system resources: {e}")


def preload():
    """
    Main preload function that orchestrates all model preloading tasks
    """
    logger.info("=" * 50)
    logger.info("STARTING MODEL PRELOADING PROCESS")
    logger.info("=" * 50)

    total_start_time = time.time()

    check_system_resources()
    optimize_tensorflow()

    if not preload_deepface_models() or not verify_model_functionality():
        logger.error("=" * 50)
        logger.error("MODEL PRELOADING FAILED")
        logger.error("=" * 50)
        exit(1)

    total_time = time.time() - total_start_time
    logger.info("=" * 50)
    logger.info(f"MODEL PRELOADING COMPLETED SUCCESSFULLY in {total_time:.2f} seconds")
    logger.info("=" * 50)


if __name__ == '__main__':
    preload()