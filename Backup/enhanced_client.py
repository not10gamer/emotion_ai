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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
CLOUD_SERVER_URL = "https://emotion-ai-227351609312.us-central1.run.app"
FRAME_SKIP = 15  # Process every 15th frame
CONNECTION_TIMEOUT = 10
CHAT_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 1  # Base delay for exponential backoff
USER_ID = str(uuid.uuid4())  # Unique user ID for conversation context

# Conversation history for local storage
conversation_history = []
# --- END CONFIGURATION ---

class ConnectionManager:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.is_connected = False
        self.last_health_check = 0
        self.health_check_interval = 30  # Check every 30 seconds
    
    def exponential_backoff_retry(self, func, max_retries=MAX_RETRIES):
        """Retry with exponential backoff"""
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
        """Check server health periodically"""
        current_time = time.time()
        if current_time - self.last_health_check > self.health_check_interval:
            try:
                response = self.session.get(f"{self.base_url}/health", timeout=5)
                response.raise_for_status()
                self.is_connected = True
                self.last_health_check = current_time
                logger.info("Health check passed")
                return True
            except Exception:
                self.is_connected = False
                logger.warning("Health check failed")
                return False
        return self.is_connected
    
    def send_emotion_request(self, frame):
        """Send frame for emotion detection with retry logic"""
        def make_request():
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            response = self.session.post(
                f"{self.base_url}/detect_emotion",
                json={'image': img_base64},
                timeout=CONNECTION_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        
        return self.exponential_backoff_retry(make_request)
    
    def send_chat_request(self, user_input, emotion, use_context=True):
        """Send chat request with retry logic"""
        def make_request():
            endpoint = "/chat/context" if use_context else "/chat"
            payload = {
                'user_input': user_input,
                'emotion': emotion
            }
            if use_context:
                payload['user_id'] = USER_ID
            
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                timeout=CHAT_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        
        return self.exponential_backoff_retry(make_request)

def send_frame_for_emotion_detection(frame, connection_manager, emotion_queue):
    """Enhanced emotion detection with better error handling"""
    try:
        result = connection_manager.send_emotion_request(frame)
        if "emotion" in result:
            emotion_info = {
                'emotion': result["emotion"],
                'confidence': result.get('confidence', 0),
                'cached': result.get('cached', False),
                'timestamp': time.time()
            }
            emotion_queue.put(emotion_info)
            
            if result.get('cached'):
                logger.debug(f"Got cached emotion: {result['emotion']}")
            else:
                logger.info(f"Detected emotion: {result['emotion']} (confidence: {result.get('confidence', 0):.2f})")
    
    except requests.exceptions.RequestException as e:
        logger.debug(f"Connection error during emotion detection: {e}")
        # Don't spam the console with connection errors
    except Exception as e:
        logger.error(f"Unexpected error in emotion detection: {e}")

def send_chat_message(user_input, current_emotion, connection_manager, chat_response_queue, use_context=True):
    """Enhanced chat with context support and error handling"""
    try:
        result = connection_manager.send_chat_request(user_input, current_emotion, use_context)
        
        if "response" in result:
            chat_info = {
                'response': result["response"],
                'emotion_context': result.get('emotion_context', current_emotion),
                'processing_time': result.get('processing_time', 0),
                'timestamp': result.get('timestamp', datetime.now().isoformat()),
                'use_context': use_context
            }
            chat_response_queue.put(chat_info)
            
            # Store in local conversation history
            conversation_entry = {
                'user_input': user_input,
                'user_emotion': current_emotion,
                'bot_response': result["response"],
                'timestamp': datetime.now().isoformat()
            }
            conversation_history.append(conversation_entry)
            
            # Keep only last 20 conversations
            if len(conversation_history) > 20:
                conversation_history.pop(0)
        else:
            chat_response_queue.put({
                'response': "Sorry, I had trouble understanding the response.",
                'error': True
            })
    
    except requests.exceptions.Timeout:
        chat_response_queue.put({
            'response': "Sorry, the request timed out. The server might be busy. Please try again.",
            'error': True
        })
    except requests.exceptions.ConnectionError:
        chat_response_queue.put({
            'response': "Sorry, I couldn't connect to the chat server. Please check your internet connection.",
            'error': True
        })
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            chat_response_queue.put({
                'response': "Sorry, you're sending messages too quickly. Please wait a moment and try again.",
                'error': True
            })
        elif e.response.status_code == 503:
            chat_response_queue.put({
                'response': "Sorry, the chatbot service is temporarily unavailable. Please try again later.",
                'error': True
            })
        else:
            chat_response_queue.put({
                'response': f"Sorry, there was a server error (HTTP {e.response.status_code}). Please try again.",
                'error': True
            })
    except Exception as e:
        logger.error(f"Unexpected chat error: {e}")
        chat_response_queue.put({
            'response': f"An unexpected error occurred: {str(e)}",
            'error': True
        })

def handle_user_input(input_queue):
    """Enhanced input handler with better error handling"""
    while True:
        try:
            user_input = input().strip()
            if user_input:  # Only process non-empty input
                input_queue.put(user_input)
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
        except (KeyboardInterrupt, EOFError):
            input_queue.put('quit')
            break
        except Exception as e:
            logger.error(f"Input error: {e}")

def save_conversation_history():
    """Save conversation history to a file"""
    try:
        filename = f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(conversation_history, f, indent=2)
        print(f"\nConversation history saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving conversation history: {e}")

def display_help():
    """Display help information"""
    print("\n--- HELP ---")
    print("Commands:")
    print("  help - Show this help message")
    print("  status - Show connection and system status")
    print("  history - Show recent conversation history")
    print("  save - Save conversation history to file")
    print("  clear - Clear screen")
    print("  quit/exit/q - Exit the application")
    print("Just type your message and press Enter to chat!")
    print("-----------\n")

def display_status(connection_manager, current_emotion, emotion_confidence):
    """Display current system status"""
    print(f"\n--- STATUS ---")
    print(f"Server URL: {CLOUD_SERVER_URL}")
    print(f"Connection: {'✓ Connected' if connection_manager.is_connected else '✗ Disconnected'}")
    print(f"User ID: {USER_ID}")
    print(f"Current Emotion: {current_emotion.capitalize()}")
    print(f"Emotion Confidence: {emotion_confidence:.2f}")
    print(f"Conversations: {len(conversation_history)}")
    print(f"Frame Skip Rate: Every {FRAME_SKIP} frames")
    print("-------------\n")

def display_recent_history(limit=5):
    """Display recent conversation history"""
    print(f"\n--- RECENT HISTORY (Last {limit}) ---")
    if not conversation_history:
        print("No conversation history yet.")
    else:
        recent = conversation_history[-limit:]
        for i, conv in enumerate(recent, 1):
            timestamp = datetime.fromisoformat(conv['timestamp']).strftime('%H:%M:%S')
            print(f"{i}. [{timestamp}] You ({conv['user_emotion']}): {conv['user_input']}")
            print(f"   AI: {conv['bot_response'][:100]}{'...' if len(conv['bot_response']) > 100 else ''}")
            print()
    print("------------------------\n")

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize connection manager
    connection_manager = ConnectionManager(CLOUD_SERVER_URL)
    
    # Initialize variables
    current_emotion = "neutral"
    emotion_confidence = 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_count = 0
    is_running = True
    last_emotion_update = 0

    # Queues for inter-thread communication
    emotion_queue = queue.Queue()
    chat_response_queue = queue.Queue()
    input_queue = queue.Queue()

    # Start the non-blocking user input thread
    input_thread = threading.Thread(target=handle_user_input, args=(input_queue,), daemon=True)
    input_thread.start()

    # Welcome message
    print("=" * 60)
    print("🤖 ENHANCED EMOTION-AWARE CHATBOT CLIENT 2.0")
    print("=" * 60)
    print(f"Connecting to: {CLOUD_SERVER_URL}")
    print(f"User ID: {USER_ID}")
    print("\n📹 OpenCV window will show your webcam feed with emotion detection")
    print("💭 Type your messages in this terminal and press Enter to chat")
    print("🎭 The AI will adapt its personality based on your detected emotion")
    print("\nAvailable commands:")
    print("  help - Show help information")
    print("  status - Show connection status")
    print("  history - Show conversation history")
    print("  save - Save conversation to file")
    print("  quit/exit/q - Exit application")
    print("\n" + "=" * 60)

    # Initial health check
    print("🔍 Checking server connection...")
    if connection_manager.health_check():
        print("✅ Connected successfully!")
    else:
        print("⚠️ Server connection failed. Will retry automatically.")

    print(f"\nYou ({current_emotion}): ", end="", flush=True)

    try:
        while is_running:
            ret, frame = cap.read()
            if not ret:
                print("\n❌ Error: Failed to capture frame from webcam.")
                break

            # Periodic health check
            connection_manager.health_check()

            # Send frame for emotion detection periodically
            if frame_count % FRAME_SKIP == 0 and connection_manager.is_connected:
                threading.Thread(
                    target=send_frame_for_emotion_detection,
                    args=(frame.copy(), connection_manager, emotion_queue),
                    daemon=True
                ).start()
            frame_count += 1

            # --- Process Queues for Non-Blocking Updates ---

            # Update emotion from queue if available
            try:
                emotion_info = emotion_queue.get_nowait()
                current_emotion = emotion_info['emotion']
                emotion_confidence = emotion_info.get('confidence', 0.0)
                last_emotion_update = time.time()
                
                # Log significant emotion changes
                if emotion_info.get('confidence', 0) > 0.8:
                    logger.info(f"High confidence emotion detected: {current_emotion}")
            except queue.Empty:
                pass

            # Check for new chat responses
            try:
                chat_info = chat_response_queue.get_nowait()
                response = chat_info['response']
                
                # Color code responses based on success/error
                if chat_info.get('error'):
                    print(f"\n❌ AI: {response}")
                else:
                    processing_time = chat_info.get('processing_time', 0)
                    context_indicator = " [with context]" if chat_info.get('use_context') else ""
                    print(f"\n🤖 AI{context_indicator}: {response}")
                    if processing_time > 0:
                        print(f"   ⏱️ Response time: {processing_time:.2f}s")
                
                print(f"\nYou ({current_emotion}): ", end="", flush=True)
            except queue.Empty:
                pass

            # Check for user input
            try:
                user_input = input_queue.get_nowait()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    is_running = False
                    break
                elif user_input.lower() == 'help':
                    display_help()
                    print(f"You ({current_emotion}): ", end="", flush=True)
                elif user_input.lower() == 'status':
                    display_status(connection_manager, current_emotion, emotion_confidence)
                    print(f"You ({current_emotion}): ", end="", flush=True)
                elif user_input.lower() == 'history':
                    display_recent_history()
                    print(f"You ({current_emotion}): ", end="", flush=True)
                elif user_input.lower() == 'save':
                    save_conversation_history()
                    print(f"You ({current_emotion}): ", end="", flush=True)
                elif user_input.lower() == 'clear':
                    import os
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(f"You ({current_emotion}): ", end="", flush=True)
                else:
                    # Regular chat message
                    if not connection_manager.is_connected:
                        print("\n⚠️ Not connected to server. Attempting to reconnect...")
                    
                    print("🤔 Thinking...")
                    
                    # Use context-aware chat by default
                    threading.Thread(
                        target=send_chat_message,
                        args=(user_input, current_emotion, connection_manager, chat_response_queue, True),
                        daemon=True
                    ).start()
            except queue.Empty:
                pass

            # --- Enhanced Video Display ---
            
            # Create a copy for drawing
            display_frame = frame.copy()
            
            # Draw emotion information
            emotion_text = f"Emotion: {current_emotion.capitalize()}"
            confidence_text = f"Confidence: {emotion_confidence:.2f}"
            connection_text = "Connected" if connection_manager.is_connected else "Disconnected"
            
            # Emotion text (green if high confidence, yellow if medium, red if low)
            if emotion_confidence > 0.8:
                emotion_color = (0, 255, 0)  # Green
            elif emotion_confidence > 0.5:
                emotion_color = (0, 255, 255)  # Yellow
            else:
                emotion_color = (0, 0, 255)  # Red
            
            cv2.putText(display_frame, emotion_text, (10, 30), font, 0.8, emotion_color, 2, cv2.LINE_AA)
            cv2.putText(display_frame, confidence_text, (10, 65), font, 0.6, emotion_color, 1, cv2.LINE_AA)
            
            # Connection status
            conn_color = (0, 255, 0) if connection_manager.is_connected else (0, 0, 255)
            cv2.putText(display_frame, f"Status: {connection_text}", (10, 100), font, 0.6, conn_color, 1, cv2.LINE_AA)
            
            # Frame rate info
            fps_text = f"Frame: {frame_count}"
            cv2.putText(display_frame, fps_text, (10, display_frame.shape[0] - 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Show frame
            cv2.imshow('Enhanced Emotion Detection', display_frame)

            # Exit on 'q' key press in video window
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                is_running = False
            elif key == ord('h'):
                display_help()
            elif key == ord('s'):
                display_status(connection_manager, current_emotion, emotion_confidence)

    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
        print(f"\n❌ Unexpected error: {e}")
    finally:
        print("\n🔄 Cleaning up...")
        
        # Save conversation history before exit
        if conversation_history:
            try:
                save_conversation_history()
            except Exception as e:
                logger.error(f"Error saving final conversation history: {e}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        connection_manager.session.close()
        
        print("👋 Application closed successfully!")
        print(f"📊 Total conversations: {len(conversation_history)}")
        print(f"🎭 Final emotion: {current_emotion}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n💥 Fatal error occurred: {e}")
        print("Please check the logs and try again.")
        sys.exit(1)