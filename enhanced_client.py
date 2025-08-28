import cv2
import requests
import base64
import threading
import queue
import time
import json
import uuid
import logging
from datetime import datetime
import sys
import speech_recognition as sr
import pygame
import io
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# --- CONFIGURATION ---
CLOUD_SERVER_URL = "https://emotion-ai-mug7a7ejoa-uc.a.run.app"
FRAME_SKIP = 15
CONNECTION_TIMEOUT = 10
CHAT_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 1
USER_ID = str(uuid.uuid4())
conversation_history = []
GRAPH_HISTORY_LENGTH = 100


# --- END CONFIGURATION ---

# --- LOGGING SETUP ---
def setup_logging():
    log_directory = "logs"
    if not os.path.exists(log_directory): os.makedirs(log_directory)
    log_filename = datetime.now().strftime(f"{log_directory}/client_%Y%m%d_%H%M%S.log")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in [file_handler, console_handler]:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    return logging.getLogger(__name__)


logger = setup_logging()
# --- END LOGGING SETUP ---

pygame.mixer.init()


class ConnectionManager:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.is_connected = False

    def exponential_backoff_retry(self, func, max_retries=MAX_RETRIES):
        for attempt in range(max_retries):
            try:
                result = func()
                self.is_connected = True
                return result
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    self.is_connected = False
                    raise e
                delay = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
                time.sleep(delay)

    def health_check(self):
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            if not self.is_connected: logger.info("Health check passed, connection re-established.")
            self.is_connected = True
            return True
        except Exception:
            if self.is_connected: logger.warning("Health check failed, connection lost.")
            self.is_connected = False
            return False

    def send_emotion_request(self, frame):
        def make_request():
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            response = self.session.post(f"{self.base_url}/detect_emotion", json={'image': img_base64},
                                         timeout=CONNECTION_TIMEOUT)
            response.raise_for_status()
            return response.json()

        return self.exponential_backoff_retry(make_request)

    def send_chat_request(self, user_input, emotion):
        def make_request():
            payload = {'user_input': user_input, 'emotion': emotion, 'user_id': USER_ID}
            response = self.session.post(f"{self.base_url}/chat/context", json=payload, timeout=CHAT_TIMEOUT)
            response.raise_for_status()
            return response.json()

        return self.exponential_backoff_retry(make_request)


def listen_for_voice(recognizer, microphone):
    if not isinstance(recognizer, sr.Recognizer): return None, "Recognizer not configured"
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("\n🎤 Listening... (Release Spacebar to send)")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            return None, "No speech detected"
    try:
        print("🤔 Recognizing...")
        return recognizer.recognize_google(audio), None
    except sr.UnknownValueError:
        return None, "Could not understand audio"
    except sr.RequestError as e:
        return None, f"Google Speech Recognition error; {e}"


def send_chat_message(user_input, current_emotion, connection_manager, chat_response_queue):
    try:
        result = connection_manager.send_chat_request(user_input, current_emotion)
        chat_response_queue.put(result)
        conversation_history.append(
            {'user_input': user_input, 'user_emotion': current_emotion, 'bot_response': result.get("response", ""),
             'timestamp': datetime.now().isoformat()})
    except Exception as e:
        chat_response_queue.put({'response': f"An unexpected error occurred: {str(e)}", 'error': True})


def play_audio_response(audio_base64):
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio_file = io.BytesIO(audio_bytes)
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): pygame.time.Clock().tick(10)
    except Exception as e:
        logger.error(f"Could not play audio response: {e}")


def update_graph(ax, fig, emotion_history):
    ax.clear()
    ax.set_ylim(0, 100)
    ax.set_title("Real-Time Emotion Confidence")
    ax.set_ylabel("Confidence (%)")
    ax.grid(True, linestyle='--', alpha=0.6)
    for emotion, data in emotion_history.items():
        if data: ax.plot(data, label=emotion.capitalize(), lw=2)
    if any(data for data in emotion_history.values()): ax.legend(loc='upper left')
    fig.canvas.draw()
    graph_img_buf = fig.canvas.tostring_argb()
    graph_img = np.frombuffer(graph_img_buf, dtype=np.uint8)
    graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    return cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    connection_manager = ConnectionManager(CLOUD_SERVER_URL)
    current_emotion, emotion_confidence = "neutral", 0.0
    frame_count = 0
    is_running = True
    emotion_request_in_flight = False

    emotion_queue = queue.Queue()
    chat_response_queue = queue.Queue()

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
    emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    emotion_history = {label: deque(maxlen=GRAPH_HISTORY_LENGTH) for label in emotion_labels}
    new_emotion_data = False

    print("=" * 60)
    print("🤖 AI TEACHER CHATBOT CLIENT 4.0 (GRAPH ENABLED)")
    print("=" * 60)
    print(f"Connecting to: {CLOUD_SERVER_URL}")
    print(f"User ID: {USER_ID}")
    print("\n📹 Two windows will open: Webcam Feed and Emotion Graph.")
    print("🎤 Press and HOLD the SPACEBAR in the video window to speak.")
    print("\n" + "=" * 60)

    logger.info("Checking server connection...")
    if not connection_manager.health_check():
        logger.warning("Server connection failed. Will retry automatically.")
    else:
        logger.info("Connected successfully!")

    print(f"\nYou ({current_emotion}): ", end="", flush=True)

    try:
        while is_running:
            ret, frame = cap.read()
            if not ret: break

            if frame_count % (FRAME_SKIP * 4) == 0:
                threading.Thread(target=connection_manager.health_check, daemon=True).start()

            if frame_count % FRAME_SKIP == 0 and connection_manager.is_connected and not emotion_request_in_flight:
                emotion_request_in_flight = True

                def detect_emotion_thread():
                    nonlocal emotion_request_in_flight, new_emotion_data
                    try:
                        result = connection_manager.send_emotion_request(frame.copy())
                        if "emotion" in result:
                            emotion_queue.put(result)
                            new_emotion_data = True
                    except Exception as e:
                        logger.debug(f"Emotion detection failed: {e}")
                    finally:
                        emotion_request_in_flight = False

                threading.Thread(target=detect_emotion_thread, daemon=True).start()
            frame_count += 1

            try:
                emotion_info = emotion_queue.get_nowait()
                current_emotion = emotion_info['emotion']
                emotion_confidence = emotion_info.get('confidence', 0.0)
                all_scores = emotion_info.get('confidence_scores', {})
                for label in emotion_labels:
                    emotion_history[label].append(all_scores.get(label, 0))
            except queue.Empty:
                pass

            try:
                chat_info = chat_response_queue.get_nowait()
                response_text = chat_info.get("response", "I'm sorry, I encountered an error.")
                print(f"\n👩‍🏫 Teacher: {response_text}")
                logger.info(f"AI Teacher: {response_text}")
                if chat_info.get("audio_response") and not chat_info.get("error"):
                    threading.Thread(target=play_audio_response, args=(chat_info["audio_response"],),
                                     daemon=True).start()
                print(f"\nYou ({current_emotion}): ", end="", flush=True)
            except queue.Empty:
                pass

            display_frame = frame.copy()
            emotion_text = f"Emotion: {current_emotion.capitalize()}"
            confidence_text = f"Confidence: {emotion_confidence:.2f}"
            status_text = "Press & Hold SPACE to Talk"
            if emotion_confidence > 0.8:
                display_color = (0, 255, 0)
            elif emotion_confidence > 0.5:
                display_color = (0, 255, 255)
            else:
                display_color = (0, 0, 255)
            cv2.putText(display_frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)
            cv2.putText(display_frame, confidence_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
            cv2.putText(display_frame, status_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow('AI Teacher', display_frame)

            if new_emotion_data:
                graph_image = update_graph(ax, fig, emotion_history)
                cv2.imshow('Emotion Graph', graph_image)
                new_emotion_data = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                is_running = False
            elif key == ord(' '):
                if not pygame.mixer.music.get_busy():
                    text, error = listen_for_voice(recognizer, microphone)
                    if error:
                        print(f"\n❌ Audio Error: {error}")
                        logger.warning(f"Audio Error: {error}")
                        print(f"\nYou ({current_emotion}): ", end="", flush=True)
                    if text:
                        print(f"\nYou ({current_emotion}): {text}")
                        logger.info(f"User (voice): {text}")
                        threading.Thread(target=send_chat_message,
                                         args=(text, current_emotion, connection_manager, chat_response_queue),
                                         daemon=True).start()
    finally:
        print("\n🔄 Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        plt.close('all')
        pygame.mixer.quit()
        logger.info("Application closed successfully!")


if __name__ == "__main__":
    main()