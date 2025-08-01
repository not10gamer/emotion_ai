import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import google.generativeai as genai
import cv2
import numpy as np
import base64
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# IMPORTANT: Use environment variables for security
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY environment variable is required")
    raise ValueError("GOOGLE_API_KEY environment variable is required")

# Emotion to Chatbot Personality Mapping
EMOTION_PROMPTS = {
    "happy": "You are a cheerful and friendly chatbot. The user is happy. Respond with encouragement and positivity.",
    "sad": "You are a gentle and empathetic chatbot. The user is sad. Respond with support, understanding, and kindness.",
    "neutral": "You are a straightforward and helpful chatbot. The user's emotion is neutral. Respond clearly and concisely.",
    "angry": "You are a calm and patient chatbot. The user is angry. Respond in a non-confrontational and de-escalating manner.",
    "surprise": "You are an enthusiastic and curious chatbot. The user is surprised. Respond with excitement and wonder.",
    "fear": "You are a reassuring and calm chatbot. The user is fearful. Respond with comfort and a sense of security.",
    "disgust": "You are a tactful and understanding chatbot. The user is disgusted. Respond with neutrality and offer to change the subject."
}
# --- END CONFIGURATION ---

# Initialize chatbot (once when the server starts)
chatbot_model = None
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    chatbot_model = genai.GenerativeModel('gemini-pro')
    logger.info("Chatbot initialized successfully on backend.")
except Exception as e:
    logger.error(f"Error initializing chatbot on backend: {e}")
    raise

# Preload DeepFace models on startup
try:
    # Warm up DeepFace with a larger dummy frame for better initialization
    dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    DeepFace.analyze(dummy_frame, actions=['emotion'], enforce_detection=False, silent=True)
    logger.info("DeepFace models preloaded successfully")
except Exception as e:
    logger.warning(f"Could not preload DeepFace models: {e}")


@app.route('/', methods=['GET'])
def root():
    """Root endpoint for basic connectivity test."""
    return jsonify({
        "message": "Emotion-Aware Chatbot API is running",
        "endpoints": ["/health", "/detect_emotion", "/chat"],
        "version": "1.0.0"
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "chatbot_ready": chatbot_model is not None
    })


@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    """Detect emotion from base64 encoded image."""
    start_time = time.time()

    if not request.json or 'image' not in request.json:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Decode base64 image
        img_bytes = base64.b64decode(request.json['image'])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Resize frame if too large (optimize performance)
        height, width = frame.shape[:2]
        if width > 640:
            ratio = 640 / width
            new_width = 640
            new_height = int(height * ratio)
            frame = cv2.resize(frame, (new_width, new_height))

        # Analyze emotion using deepface
        analysis = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )

        dominant_emotion = "neutral"
        confidence_scores = {}

        if analysis and isinstance(analysis, list) and len(analysis) > 0:
            dominant_emotion = analysis[0]['dominant_emotion']
            confidence_scores = analysis[0].get('emotion', {})

        processing_time = time.time() - start_time
        logger.info(f"Emotion detection completed in {processing_time:.2f}s: {dominant_emotion}")

        return jsonify({
            "emotion": dominant_emotion,
            "confidence_scores": confidence_scores,
            "processing_time": processing_time
        })

    except Exception as e:
        logger.error(f"Error during emotion detection: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Generate chatbot response based on user input and emotion."""
    start_time = time.time()

    if not request.json:
        return jsonify({"error": "No JSON data provided"}), 400

    user_input = request.json.get('user_input')
    emotion = request.json.get('emotion', 'neutral')

    if not user_input or not user_input.strip():
        return jsonify({"error": "No user input provided"}), 400

    if not chatbot_model:
        return jsonify({"error": "Chatbot not initialized on server"}), 500

    try:
        system_prompt = EMOTION_PROMPTS.get(emotion, EMOTION_PROMPTS["neutral"])
        full_prompt = f"{system_prompt}\n\nUser: {user_input.strip()}\nAI:"

        response = chatbot_model.generate_content(full_prompt)

        processing_time = time.time() - start_time
        logger.info(f"Chat response generated in {processing_time:.2f}s for emotion: {emotion}")

        return jsonify({
            "response": response.text,
            "emotion_context": emotion,
            "processing_time": processing_time
        })

    except Exception as e:
        logger.error(f"Error during chatbot interaction: {e}")
        return jsonify({"error": "Failed to generate response"}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # For local development only
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)