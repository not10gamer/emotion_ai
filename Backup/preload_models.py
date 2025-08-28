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
        # Set memory growth to avoid GPU memory issues
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Configured {len(gpus)} GPU(s) with memory growth")
        else:
            logger.info("No GPUs found, using CPU")
        
        # Set threading options for better CPU performance
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
        
        # Disable unnecessary TensorFlow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
        
        logger.info("TensorFlow optimizations applied successfully")
        return True
    except Exception as e:
        logger.error(f"Error optimizing TensorFlow: {e}")
        return False

def preload_deepface_models():
    """
    Preloads all DeepFace models to avoid cold start delays and race conditions
    when running with multiple gunicorn workers.
    """
    try:
        logger.info("Starting DeepFace model preloading...")
        start_time = time.time()
        
        # Create dummy frame with proper dimensions
        dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Add some variation to the dummy frame to trigger proper model loading
        dummy_frame[50:150, 50:150] = [128, 128, 128]  # Gray square
        dummy_frame[100:120, 100:120] = [255, 255, 255]  # White square (simulated face)
        
        # Preload emotion detection model
        logger.info("Preloading emotion detection model...")
        result = DeepFace.analyze(
            dummy_frame, 
            actions=['emotion'], 
            enforce_detection=False, 
            silent=True
        )
        logger.info(f"Emotion model loaded. Sample result: {result[0]['dominant_emotion'] if result else 'None'}")
        
        # Test with multiple dummy frames to ensure model stability
        logger.info("Testing model stability with multiple frames...")
        for i in range(3):
            test_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            DeepFace.analyze(test_frame, actions=['emotion'], enforce_detection=False, silent=True)
        
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
        
        # Create a more realistic test image
        test_frame = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        # Add some face-like features
        # Simulated eyes
        test_frame[80:90, 70:80] = [0, 0, 0]
        test_frame[80:90, 140:150] = [0, 0, 0]
        
        # Simulated mouth
        test_frame[150:160, 100:120] = [100, 50, 50]
        
        result = DeepFace.analyze(
            test_frame,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        
        if result and len(result) > 0:
            emotions = result[0].get('emotion', {})
            dominant_emotion = result[0].get('dominant_emotion', 'unknown')
            logger.info(f"Model verification successful. Detected emotion: {dominant_emotion}")
            logger.info(f"Emotion scores: {emotions}")
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
        
        logger.info(f"System Resources:")
        logger.info(f"  CPU cores: {cpu_count}")
        logger.info(f"  Total memory: {memory.total / (1024**3):.2f} GB")
        logger.info(f"  Available memory: {memory.available / (1024**3):.2f} GB")
        logger.info(f"  Memory usage: {memory.percent}%")
        
        # Check if we have enough memory for the models
        if memory.available < 1024**3:  # Less than 1GB available
            logger.warning("Low memory available. Performance may be affected.")
        
        return True
    except ImportError:
        logger.info("psutil not available, skipping resource check")
        return True
    except Exception as e:
        logger.error(f"Error checking system resources: {e}")
        return True  # Don't fail the preload for this

def preload():
    """
    Main preload function that orchestrates all model preloading tasks
    """
    logger.info("="*50)
    logger.info("STARTING MODEL PRELOADING PROCESS")
    logger.info("="*50)
    
    total_start_time = time.time()
    success = True
    
    # Check system resources
    logger.info("Step 1: Checking system resources...")
    check_system_resources()
    
    # Optimize TensorFlow
    logger.info("Step 2: Optimizing TensorFlow settings...")
    if not optimize_tensorflow():
        logger.warning("TensorFlow optimization failed, but continuing...")
    
    # Preload DeepFace models
    logger.info("Step 3: Preloading DeepFace models...")
    if not preload_deepface_models():
        logger.error("Failed to preload DeepFace models")
        success = False
    
    # Verify functionality
    logger.info("Step 4: Verifying model functionality...")
    if not verify_model_functionality():
        logger.error("Model verification failed")
        success = False
    
    total_time = time.time() - total_start_time
    
    if success:
        logger.info("="*50)
        logger.info(f"MODEL PRELOADING COMPLETED SUCCESSFULLY in {total_time:.2f} seconds")
        logger.info("="*50)
    else:
        logger.error("="*50)
        logger.error(f"MODEL PRELOADING FAILED after {total_time:.2f} seconds")
        logger.error("="*50)
        # Exit with error code for Docker build to fail
        exit(1)

if __name__ == '__main__':
    preload()