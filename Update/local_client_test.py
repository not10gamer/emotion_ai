import subprocess
import shlex

CLOUD_SERVER_URL = "https://emotion-chatbot-227351609312.asia-southeast2.run.app"


def test_connection(server_url):
    """Tests the connection to the server using curl."""
    print(f"--- Testing Connection to: {server_url} ---")
    command = f"curl -X POST -H \"Content-Type: application/json\" -d '{{\"user_input\": \"hello\", \"emotion\": \"neutral\"}}' {server_url}/chat"
    print(f"Running command: {command}")
    try:
        process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=60)
        print("\n--- Response ---")
        if process.returncode == 0:
            print("Connection successful!")
            print("Response from server:")
            print(stdout)
        else:
            print("Connection failed.")
            print(f"Return Code: {process.returncode}")
            print("\n--- STDOUT ---")
            print(stdout)
            print("\n--- STDERR ---")
            print(stderr)
    except subprocess.TimeoutExpired:
        print("Connection timed out. The server is taking too long to respond.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_connection(CLOUD_SERVER_URL)
