import os
import logging
import time
import hashlib
import functools
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from deepface import DeepFace
import google.generativeai as genai
import cv2
import numpy as np
import base64
import redis
import psutil
from werkzeug.exceptions import HTTPException
import threading
import queue

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"  # Use Redis in production
)

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2097152"))  # 2MB
EMOTION_CONFIDENCE_THRESHOLD = float(os.getenv("EMOTION_CONFIDENCE_THRESHOLD", "0.6"))

if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY environment variable is required")
    raise ValueError("GOOGLE_API_KEY environment variable is required")

# Enhanced emotion prompts with more personality
EMOTION_PROMPTS = {
    "happy": "You are an enthusiastic and joyful chatbot. The user is happy! Match their positive energy with excitement, encouragement, and celebratory responses. Use upbeat language and be genuinely excited for them.",
    "sad": "You are a compassionate and gentle chatbot. The user seems sad. Respond with deep empathy, offer comfort, and provide emotional support. Be a good listener and offer hope without dismissing their feelings.",
    "neutral": "You are a balanced and helpful chatbot. The user's emotion is neutral. Provide clear, informative responses while being friendly and approachable. Keep the conversation engaging but not overwhelming.",
    "angry": "You are a calm and understanding chatbot. The user appears angry or frustrated. Respond with patience, validate their feelings, and help de-escalate the situation. Avoid being confrontational and offer constructive solutions.",
    "surprise": "You are an enthusiastic and curious chatbot. The user seems surprised! Share in their amazement and wonder. Ask engaging questions and explore their surprise with genuine interest and excitement.",
    "fear": "You are a reassuring and protective chatbot. The user appears fearful or anxious. Provide comfort, security, and calm reassurance. Help them feel safe and offer practical support to address their concerns.",
    "disgust": "You are a tactful and understanding chatbot. The user seems disgusted or uncomfortable. Respond with empathy while being respectful of their feelings. Offer to change topics or provide alternative perspectives gently."
}

# Global variables
chatbot_model = None
start_time = time.time()
request_counter = 0
redis_client = None

# Conversation memory
class ConversationManager:
    def __init__(self):
        self.conversations = {}
        self.lock = threading.Lock()
    
    def add_context(self, user_id, message, emotion, response):
        with self.lock:
            if user_id not in self.conversations:
                self.conversations[user_id] = []
            
            self.conversations[user_id].append({
                'user_message': message,
                'emotion': emotion,
                'bot_response': response,
                'timestamp': time.time()
            })
            
            # Keep only last 10 conversations per user
            if len(self.conversations[user_id]) > 10:
                self.conversations[user_id] = self.conversations[user_id][-10:]
    
    def get_context_prompt(self, user_id, current_emotion):
        with self.lock:
            context = self.conversations.get(user_id, [])[-3:]  # Last 3 exchanges
            if not context:
                return ""
            
            context_str = "Previous conversation context:\n"
            for conv in context:
                context_str += f"User ({conv['emotion']}): {conv['user_message']}\n"
                context_str += f"AI: {conv['bot_response']}\n"
            
            return context_str

conversation_manager = ConversationManager()

# Initialize Redis for caching
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connection established")
except Exception as e:
    logger.warning(f"Redis not available, using in-memory caching: {e}")
    redis_client = None

@functools.lru_cache(maxsize=100)
def get_cached_emotion(image_hash, timestamp_hour):
    """Cache emotion results for similar images within the same hour"""
    if redis_client:
        try:
            cache_key = f"emotion:{image_hash}:{timestamp_hour}"
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return eval(cached_result)  # Note: Use json.loads in production
        except Exception as e:
            logger.warning(f"Redis cache read error: {e}")
    return None

def cache_emotion_result(image_hash, timestamp_hour, result):
    """Cache emotion detection result"""
    if redis_client:
        try:
            cache_key = f"emotion:{image_hash}:{timestamp_hour}"
            redis_client.setex(cache_key, 3600, str(result))  # Cache for 1 hour
        except Exception as e:
            logger.warning(f"Redis cache write error: {e}")

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return {
        "memory_percent": process.memory_percent(),
        "memory_info": process.memory_info()._asdict()
    }

def validate_image_data(image_data):
    """Validate image data before processing"""
    if not image_data:
        return False, "No image data provided"
    
    try:
        img_bytes = base64.b64decode(image_data)
        if len(img_bytes) > MAX_IMAGE_SIZE:
            return False, f"Image too large. Max size: {MAX_IMAGE_SIZE} bytes"
        
        # Try to decode the image
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False, "Invalid image format"
        
        return True, frame
    except Exception as e:
        return False, f"Image validation error: {str(e)}"

def exponential_backoff_retry(func, max_retries=3, base_delay=1):
    """Retry function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
            logger.warning(f"Retry attempt {attempt + 1} after {delay}s delay")

# Initialize chatbot model
def initialize_chatbot():
    global chatbot_model
    try:
        logger.info("Initializing Google Generative AI...")
        genai.configure(api_key=GOOGLE_API_KEY)
        chatbot_model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Chatbot initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}", exc_info=True)
        return False

# Preload models
def preload_models():
    try:
        logger.info("Preloading DeepFace models...")
        dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.analyze(dummy_frame, actions=['emotion'], enforce_detection=False, silent=True)
        logger.info("DeepFace models preloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Could not preload DeepFace models: {e}")
        return False

# Initialize everything
chatbot_ready = initialize_chatbot()
deepface_ready = preload_models()

@app.before_request
def before_request():
    global request_counter
    request_counter += 1
    g.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - g.start_time
    logger.info(f"Request processed in {duration:.3f}s - {request.method} {request.path} - {response.status_code}")
    return response

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with comprehensive API information."""
    return jsonify({
        "message": "Enhanced Emotion-Aware Chatbot API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "/health": "Health check and system status",
            "/metrics": "System metrics and statistics", 
            "/detect_emotion": "POST - Detect emotion from image",
            "/chat": "POST - Chat with emotion-aware AI",
            "/chat/context": "POST - Chat with conversation context"
        },
        "rate_limits": {
            "default": "200 per day, 50 per hour",
            "chat": "10 per minute",
            "emotion_detection": "30 per minute"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - start_time,
        "services": {
            "chatbot": chatbot_model is not None,
            "deepface": deepface_ready,
            "redis": redis_client is not None
        },
        "memory": get_memory_usage(),
        "requests_processed": request_counter
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    """System metrics endpoint for monitoring."""
    return jsonify({
        "uptime": time.time() - start_time,
        "requests_processed": request_counter,
        "memory_usage": get_memory_usage(),
        "cache_stats": {
            "redis_available": redis_client is not None,
            "lru_cache_info": get_cached_emotion.cache_info()._asdict() if hasattr(get_cached_emotion, 'cache_info') else {}
        },
        "model_status": {
            "chatbot_ready": chatbot_model is not None,
            "deepface_ready": deepface_ready
        }
    })

@app.route('/detect_emotion', methods=['POST'])
@limiter.limit("30 per minute")
def detect_emotion():
    """Enhanced emotion detection with caching and validation."""
    start_time = time.time()

    if not request.json or 'image' not in request.json:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Validate image
        is_valid, result = validate_image_data(request.json['image'])
        if not is_valid:
            return jsonify({"error": result}), 400
        
        frame = result
        
        # Generate image hash for caching
        img_bytes = base64.b64decode(request.json['image'])
        image_hash = hashlib.md5(img_bytes).hexdigest()
        current_hour = datetime.now().strftime("%Y%m%d%H")
        
        # Check cache first
        cached_result = get_cached_emotion(image_hash, current_hour)
        if cached_result:
            logger.info(f"Returning cached emotion result for {image_hash}")
            return jsonify({
                **cached_result,
                "cached": True,
                "processing_time": time.time() - start_time
            })

        # Resize frame for better performance
        height, width = frame.shape[:2]
        if width > 640:
            ratio = 640 / width
            new_width = 640
            new_height = int(height * ratio)
            frame = cv2.resize(frame, (new_width, new_height))

        # Analyze emotion with retry logic
        def analyze_emotion():
            return DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

        analysis = exponential_backoff_retry(analyze_emotion)

        dominant_emotion = "neutral"
        confidence_scores = {}
        confidence = 0

        if analysis and isinstance(analysis, list) and len(analysis) > 0:
            emotion_data = analysis[0]
            dominant_emotion_str = emotion_data['dominant_emotion']
            confidence = emotion_data['emotion'][dominant_emotion_str] / 100.0  # Convert to 0-1 range
            
            logger.info(f"Detected emotion: {dominant_emotion_str} (confidence: {confidence:.2f})")

            # Use emotion only if confidence is above threshold
            if confidence >= EMOTION_CONFIDENCE_THRESHOLD:
                dominant_emotion = dominant_emotion_str
                confidence_scores = emotion_data.get('emotion', {})
                logger.info(f"Confidence threshold met. Using emotion: {dominant_emotion}")
            else:
                logger.info(f"Low confidence ({confidence:.2f}). Using neutral emotion.")
                dominant_emotion = "neutral"

        processing_time = time.time() - start_time
        
        result = {
            "emotion": dominant_emotion,
            "confidence": confidence,
            "confidence_scores": confidence_scores,
            "processing_time": processing_time,
            "cached": False
        }
        
        # Cache the result
        cache_emotion_result(image_hash, current_hour, result)
        
        logger.info(f"Emotion detection completed in {processing_time:.2f}s: {dominant_emotion}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error during emotion detection: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during emotion detection"}), 500

@app.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    """Enhanced chat endpoint with better error handling."""
    logger.info("=== CHAT ENDPOINT START ===")
    start_time = time.time()

    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400

        user_input = request.json.get('user_input', '').strip()
        emotion = request.json.get('emotion', 'neutral')
        
        # Input validation
        if not user_input:
            return jsonify({"error": "No user input provided"}), 400
        
        if len(user_input) > 1000:
            return jsonify({"error": "Input too long. Maximum 1000 characters."}), 400

        if not chatbot_model:
            return jsonify({"error": "Chatbot service unavailable"}), 503

        # Get system prompt
        system_prompt = EMOTION_PROMPTS.get(emotion.lower(), EMOTION_PROMPTS["neutral"])
        full_prompt = f"{system_prompt}\n\nUser: {user_input}\nAI:"

        logger.info(f"Processing chat request - Emotion: {emotion}, Input length: {len(user_input)}")

        # Generate response with retry logic
        def generate_response():
            return chatbot_model.generate_content(full_prompt)

        response = exponential_backoff_retry(generate_response)
        
        if not response or not response.text:
            return jsonify({"error": "Failed to generate response"}), 500

        processing_time = time.time() - start_time
        
        result = {
            "response": response.text.strip(),
            "emotion_context": emotion,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Chat completed successfully in {processing_time:.2f}s")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        return jsonify({"error": "Failed to process chat request"}), 500

@app.route('/chat/context', methods=['POST'])
@limiter.limit("10 per minute")
def chat_with_context():
    """Chat endpoint that maintains conversation context."""
    start_time = time.time()

    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400

        user_input = request.json.get('user_input', '').strip()
        emotion = request.json.get('emotion', 'neutral')
        user_id = request.json.get('user_id', 'anonymous')
        
        if not user_input:
            return jsonify({"error": "No user input provided"}), 400

        if not chatbot_model:
            return jsonify({"error": "Chatbot service unavailable"}), 503

        # Get conversation context
        context_prompt = conversation_manager.get_context_prompt(user_id, emotion)
        system_prompt = EMOTION_PROMPTS.get(emotion.lower(), EMOTION_PROMPTS["neutral"])
        
        full_prompt = f"{system_prompt}\n\n{context_prompt}\nCurrent user message ({emotion}): {user_input}\nAI:"

        def generate_response():
            return chatbot_model.generate_content(full_prompt)

        response = exponential_backoff_retry(generate_response)
        
        if not response or not response.text:
            return jsonify({"error": "Failed to generate response"}), 500

        bot_response = response.text.strip()
        
        # Store conversation context
        conversation_manager.add_context(user_id, user_input, emotion, bot_response)

        processing_time = time.time() - start_time
        
        return jsonify({
            "response": bot_response,
            "emotion_context": emotion,
            "user_id": user_id,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Context chat error: {e}", exc_info=True)
        return jsonify({"error": "Failed to process contextual chat request"}), 500

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    return jsonify({"error": e.description}), e.code

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again later.",
        "retry_after": e.retry_after
    }), 429

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    logger.info(f"Chatbot ready: {chatbot_ready}")
    logger.info(f"DeepFace ready: {deepface_ready}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)