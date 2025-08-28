import os
import logging
import time
from datetime import datetime
import base64
import io
import threading
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from deepface import DeepFace
import google.generativeai as genai
import cv2
import numpy as np
from gtts import gTTS
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"],
                  storage_uri="memory://")

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2097152"))
EMOTION_CONFIDENCE_THRESHOLD = float(os.getenv("EMOTION_CONFIDENCE_THRESHOLD", "0.5"))

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

# --- AI & Model Setup ---
genai.configure(api_key=GOOGLE_API_KEY)
chatbot_model = genai.GenerativeModel('gemini-1.5-flash')
EMOTION_PROMPTS = {
    "happy": "You are an AI teacher. The student is happy and engaged! As their teacher, reinforce their success with positive encouragement and enthusiasm.",
    "sad": "You are a compassionate AI teacher. The student seems sad or discouraged. Respond with empathy and gentle support.",
    "neutral": "You are a knowledgeable AI teacher. The student's emotion is neutral. Spark their curiosity by providing clear explanations and asking insightful questions.",
    "angry": "You are a calm and patient AI teacher. The student appears frustrated. Validate their feelings and help de-escalate the situation.",
    "surprise": "You are an inspiring AI teacher. The student is surprised! Share in their moment of discovery and encourage their curiosity.",
    "fear": "You are a reassuring AI teacher. The student seems anxious or fearful. Provide comfort and calm reassurance.",
    "disgust": "You are a tactful AI teacher. The student seems uncomfortable. Respect their feelings and gently offer to move on."
}


class ConversationManager:
    def __init__(self):
        self.conversations = {}
        self.lock = threading.Lock()

    def add_context(self, user_id, message, emotion, response):
        with self.lock:
            if user_id not in self.conversations: self.conversations[user_id] = []
            self.conversations[user_id].append({'user_message': message, 'emotion': emotion, 'bot_response': response})
            self.conversations[user_id] = self.conversations[user_id][-10:]

    def get_context_prompt(self, user_id):
        with self.lock:
            context = self.conversations.get(user_id, [])[-3:]
            if not context: return ""
            context_str = "Previous conversation context:\n"
            for conv in context:
                context_str += f"Student ({conv['emotion']}): {conv['user_message']}\nTeacher: {conv['bot_response']}\n"
            return context_str


conversation_manager = ConversationManager()


# --- Preloading Models at Startup ---
def preload_models():
    try:
        dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.analyze(dummy_frame, actions=['emotion'], enforce_detection=False, silent=True,
                         detector_backend='mtcnn')
        logger.info("DeepFace models (mtcnn) preloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Could not preload DeepFace models: {e}")
        return False


preload_models()


# --- End Preloading ---

@app.before_request
def before_request_hook():
    g.start_time = time.time()


@app.after_request
def after_request_hook(response):
    duration = time.time() - g.start_time
    logger.info(f"Request processed in {duration:.3f}s - {request.method} {request.path} - {response.status_code}")
    return response


# --- API Endpoints ---
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"})


@app.route('/detect_emotion', methods=['POST'])
@limiter.limit("30 per minute")
def detect_emotion():
    try:
        image_data = request.json.get('image')
        if not image_data: return jsonify({"error": "No image data provided"}), 400

        img_bytes = base64.b64decode(image_data)
        if len(img_bytes) > MAX_IMAGE_SIZE: return jsonify({"error": "Image too large"}), 400
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None: return jsonify({"error": "Invalid image format"}), 400

        dominant_emotion, confidence, confidence_scores = "neutral", 0, {}

        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True, silent=True,
                                        detector_backend='mtcnn')
            if analysis and isinstance(analysis, list) and len(analysis) > 0:
                emotion_data = analysis[0]
                dominant_emotion_str = emotion_data['dominant_emotion']
                confidence = emotion_data['emotion'][dominant_emotion_str] / 100.0
                if confidence >= EMOTION_CONFIDENCE_THRESHOLD:
                    dominant_emotion = dominant_emotion_str
                confidence_scores = emotion_data.get('emotion', {})
        except ValueError:
            logger.debug("No face detected in frame.")

        return jsonify({"emotion": dominant_emotion, "confidence": confidence, "confidence_scores": confidence_scores})
    except Exception as e:
        logger.error(f"Error during emotion detection: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during emotion detection"}), 500


@app.route('/chat/context', methods=['POST'])
@limiter.limit("10 per minute")
def chat_with_context():
    try:
        data = request.json
        if not data: return jsonify({"error": "No JSON data provided"}), 400
        user_input, emotion, user_id = data.get('user_input', '').strip(), data.get('emotion', 'neutral'), data.get(
            'user_id')
        if not user_input or not user_id: return jsonify({"error": "User input and user ID are required"}), 400

        context_prompt = conversation_manager.get_context_prompt(user_id)
        system_prompt = EMOTION_PROMPTS.get(emotion, EMOTION_PROMPTS["neutral"])
        full_prompt = f"{system_prompt}\n\n{context_prompt}\nCurrent student message ({emotion}): {user_input}\nTeacher:"

        response = chatbot_model.generate_content(full_prompt)
        bot_response = response.text.strip()
        conversation_manager.add_context(user_id, user_input, emotion, bot_response)

        audio_fp = io.BytesIO()
        tts = gTTS(text=bot_response, lang='en')
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio_base64 = base64.b64encode(audio_fp.read()).decode('utf-8')

        return jsonify({"response": bot_response, "audio_response": audio_base64})
    except Exception as e:
        logger.error(f"Context chat error: {e}", exc_info=True)
        return jsonify({"error": "Failed to process chat request"}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException): return jsonify(error=e.description), e.code
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify(error="Internal server error"), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)