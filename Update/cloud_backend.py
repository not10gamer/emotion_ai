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
    logger.info("Initializing Google Generative AI...")
    genai.configure(api_key=GOOGLE_API_KEY)
    chatbot_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Chatbot initialized successfully on backend.")
except Exception as e:
    logger.error(f"Error initializing chatbot on backend: {e}", exc_info=True)
    chatbot_model = None # Ensure model is None if init fails

# Preload DeepFace models on startup
try:
    logger.info("Preloading DeepFace models...")
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
        img_bytes = base64.b64decode(request.json['image'])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

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
            emotion_data = analysis[0]
            dominant_emotion_str = emotion_data['dominant_emotion']
            confidence = emotion_data['emotion'][dominant_emotion_str]
            
            logger.info(f"Detected dominant emotion: {dominant_emotion_str} with confidence: {confidence:.2f}%")

            # Only accept the emotion if the model is reasonably confident
            if confidence > 75:
                dominant_emotion = dominant_emotion_str
                confidence_scores = emotion_data.get('emotion', {})
                logger.info(f"Confidence threshold passed. Using emotion: {dominant_emotion}")
            else:
                logger.info(f"Confidence threshold not met. Defaulting to neutral.")
                dominant_emotion = "neutral"

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
    logger.info("--- CHAT ENDPOINT START ---")
    start_time = time.time()

    try:
        logger.info("Step 1: Checking for JSON data...")
        if not request.json:
            logger.error("No JSON data provided.")
            return jsonify({"error": "No JSON data provided"}), 400
        logger.info("Step 1: JSON data found.")

        logger.info("Step 2: Extracting user_input and emotion...")
        user_input = request.json.get('user_input')
        emotion = request.json.get('emotion', 'neutral')
        logger.info(f"Step 2: user_input='{user_input}', emotion='{emotion}'")

        logger.info("Step 3: Validating user_input...")
        if not user_input or not user_input.strip():
            logger.error("User input is empty or missing.")
            return jsonify({"error": "No user input provided"}), 400
        logger.info("Step 3: User input is valid.")

        logger.info("Step 4: Checking if chatbot model is initialized...")
        if not chatbot_model:
            logger.error("Chatbot model is not initialized.")
            return jsonify({"error": "Chatbot not initialized on server"}), 500
        logger.info("Step 4: Chatbot model is initialized.")

        logger.info("Step 5: Getting system prompt...")
        system_prompt = EMOTION_PROMPTS.get(emotion, EMOTION_PROMPTS["neutral"])
        full_prompt = f"{system_prompt}\n\nUser: {user_input.strip()}\nAI:"
        logger.info("Step 5: System prompt created.")

        logger.info("Step 6: Calling generate_content on the model...")
        response = chatbot_model.generate_content(full_prompt)
        logger.info("Step 6: Successfully received response from model.")

        logger.info("Step 7: Preparing final JSON response...")
        processing_time = time.time() - start_time
        final_response = jsonify({
            "response": response.text,
            "emotion_context": emotion,
            "processing_time": processing_time
        })
        logger.info(f"Step 7: JSON response prepared. Total time: {processing_time:.2f}s")
        logger.info("--- CHAT ENDPOINT SUCCESS ---")
        return final_response

    except Exception as e:
        logger.error(f"--- CHAT ENDPOINT FAILED ---: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate response: {e}"}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
