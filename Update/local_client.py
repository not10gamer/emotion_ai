import cv2
import requests
import base64
import threading
import queue
import time

# --- CONFIGURATION ---
# IMPORTANT: Replace with the actual URL of your deployed cloud backend
CLOUD_SERVER_URL = "https://emotion-ai-mug7a7ejoa-uc.a.run.app" # For local testing, use localhost. For cloud, use your server's public IP/domain

# How often to send frames for emotion detection (e.g., 15 means twice per second at 30 FPS)
FRAME_SKIP = 15
# --- END CONFIGURATION ---

def send_frame_for_emotion_detection(frame, server_url, emotion_queue):
    """Sends a frame to the cloud backend for emotion detection."""
    try:
        # Encode frame to JPEG and then to base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        response = requests.post(f"{server_url}/detect_emotion", json={'image': img_base64}, timeout=5)
        response.raise_for_status()
        result = response.json()
        if "emotion" in result:
            emotion_queue.put(result["emotion"])
    except requests.exceptions.RequestException:
        # Silently ignore connection errors to prevent console spam
        pass
    except Exception as e:
        print(f"\n[Error in emotion thread: {e}]")

def send_chat_message(user_input, current_emotion, server_url, chat_response_queue):
    """Sends user input and emotion to the cloud backend for chatbot response."""
    try:
        response = requests.post(
            f"{server_url}/chat",
            json={'user_input': user_input, 'emotion': current_emotion},
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        if "response" in result:
            chat_response_queue.put(result["response"])
        else:
            chat_response_queue.put("Sorry, I had trouble understanding the response.")
    except requests.exceptions.RequestException as e:
        chat_response_queue.put(f"Sorry, I couldn't connect to the chat server. Reason: {e}")
    except Exception as e:
        chat_response_queue.put(f"An unexpected chat error occurred: {e}")

def handle_user_input(input_queue):
    """Handles user input in a separate thread to prevent blocking the main UI."""
    while True:
        try:
            user_input = input()
            input_queue.put(user_input)
            if user_input.lower() in ['quit', 'exit']:
                break
        except (KeyboardInterrupt, EOFError):
            input_queue.put('quit')
            break

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    current_emotion = "neutral"
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_count = 0
    is_running = True

    # Queues for inter-thread communication
    emotion_queue = queue.Queue()
    chat_response_queue = queue.Queue()
    input_queue = queue.Queue()

    # Start the non-blocking user input thread
    input_thread = threading.Thread(target=handle_user_input, args=(input_queue,), daemon=True)
    input_thread.start()

    print("--- Cloud-Powered Emotion Aware Chatbot (Local Client) ---")
    print(f"Connecting to cloud backend at: {CLOUD_SERVER_URL}")
    print("An OpenCV window will show your webcam feed.")
    print("Your detected emotion will be displayed on the video.")
    print("Type your message in this terminal and press Enter.")
    print("Type 'quit' or 'exit' to close the application.\n")
    print(f"You ({current_emotion}): ", end="", flush=True)

    while is_running:
        ret, frame = cap.read()
        if not ret:
            print("\nError: Failed to capture frame.")
            break

        # Send frame for emotion detection periodically
        if frame_count % FRAME_SKIP == 0:
            threading.Thread(
                target=send_frame_for_emotion_detection,
                args=(frame.copy(), CLOUD_SERVER_URL, emotion_queue),
                daemon=True
            ).start()
        frame_count += 1

        # --- Check Queues for Non-Blocking Updates ---

        # Update emotion from queue if available
        try:
            current_emotion = emotion_queue.get_nowait()
        except queue.Empty:
            pass

        # Check for new chat responses from the cloud
        try:
            bot_response = chat_response_queue.get_nowait()
            print(f"\nAI: {bot_response}")
            print(f"You ({current_emotion}): ", end="", flush=True)
        except queue.Empty:
            pass

        # Check for user input from the input thread
        try:
            user_input = input_queue.get_nowait()
            # Newline is handled by the input() function itself
            if user_input.lower() in ['quit', 'exit']:
                is_running = False
                break

            print("Thinking...")
            threading.Thread(
                target=send_chat_message,
                args=(user_input, current_emotion, CLOUD_SERVER_URL, chat_response_queue),
                daemon=True
            ).start()
        except queue.Empty:
            pass

        # --- Display UI ---
        emotion_text = f"Emotion: {current_emotion.capitalize()}"
        cv2.putText(frame, emotion_text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Emotion Detection (Cloud Powered)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False

    print("\nExiting...")
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()