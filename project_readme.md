# 🤖 Enhanced Emotion-Aware Chatbot

An advanced AI-powered chatbot that adapts its personality and responses based on real-time facial emotion detection. Built with Python, Flask, OpenCV, and Google's Generative AI, deployed on Google Cloud Run.

[![Cloud Run](https://img.shields.io/badge/Google%20Cloud-Run-blue?logo=google-cloud)](https://cloud.google.com/run)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)]()

## ✨ Features

### Core Functionality
- 🎭 **Real-time Emotion Detection**: Advanced facial emotion recognition using DeepFace
- 🤖 **Adaptive AI Responses**: Google Generative AI with emotion-aware personalities
- 💬 **Conversation Context**: Maintains conversation history for natural interactions
- ⚡ **High Performance**: Optimized with caching, rate limiting, and auto-scaling

### Technical Features
- 🔄 **Auto-reconnection**: Exponential backoff retry logic
- 📊 **Comprehensive Monitoring**: Health checks, metrics, and structured logging
- 🛡️ **Security**: Rate limiting, input validation, and error handling
- 🎯 **Testing Suite**: Automated API testing with detailed reporting
- 📱 **Cross-platform Client**: Works on Windows, macOS, and Linux

### Cloud-Native Architecture
- ☁️ **Google Cloud Run**: Serverless, auto-scaling deployment
- 🗄️ **Redis Caching**: Optional caching layer for improved performance
- 🐳 **Docker**: Containerized application with multi-stage builds
- 🔐 **Security**: Non-root containers, secrets management

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Required tools
- Google Cloud SDK
- Docker
- Python 3.11+
- Webcam (for client)
```

### 2. Deploy to Google Cloud

```bash
# Set up your project
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com

# Set your API key
gcloud secrets create GOOGLE_API_KEY --data-file=- <<< "your-google-ai-api-key"

# Deploy
gcloud builds submit --config cloudbuild.yaml .
```

### 3. Run the Client

```bash
# Install dependencies
pip install opencv-python requests

# Update server URL in local_client.py
CLOUD_SERVER_URL = "https://your-deployed-service-url"

# Launch the interactive client
python enhanced_local_client.py
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Local Client  │───▶│   Cloud Backend  │───▶│  Google AI API  │
│   (OpenCV +     │    │  (Flask + ML)    │    │   (Gemini)      │
│    Webcam)      │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       
         │                       ▼                       
         │              ┌──────────────────┐             
         │              │   DeepFace ML    │             
         │              │ (Emotion Model)  │             
         │              └──────────────────┘             
         │                       │                       
         ▼                       ▼                       
┌─────────────────┐    ┌──────────────────┐             
│ Conversation    │    │  Redis Cache     │             
│ History (Local) │    │  (Optional)      │             
└─────────────────┘    └──────────────────┘             
```

## 📊 API Endpoints

### Core Endpoints

| Endpoint | Method | Description | Rate Limit |
|----------|--------|-------------|------------|
| `/` | GET | API information | - |
| `/health` | GET | Health check and status | - |
| `/metrics` | GET | System metrics | - |
| `/detect_emotion` | POST | Analyze facial emotion | 30/min |
| `/chat` | POST | Basic chat interaction | 10/min |
| `/chat/context` | POST | Context-aware chat | 10/min |

### Request Examples

**Emotion Detection**:
```json
POST /detect_emotion
{
  "image": "base64-encoded-image-data"
}
```

**Chat**:
```json
POST /chat
{
  "user_input": "How are you today?",
  "emotion": "happy"
}
```

**Context Chat**:
```json
POST /chat/context
{
  "user_input": "What did we discuss?",
  "emotion": "neutral",
  "user_id": "unique-user-id"
}
```

## 🎭 Emotion-Based Personalities

The AI adapts its responses based on detected emotions:

- **😊 Happy**: Enthusiastic, encouraging, celebratory
- **😢 Sad**: Gentle, empathetic, supportive
- **😐 Neutral**: Balanced, informative, friendly
- **😠 Angry**: Calm, patient, de-escalating
- **😲 Surprised**: Curious, enthusiastic, wondering
- **😨 Fearful**: Reassuring, comforting, protective
- **🤢 Disgusted**: Tactful, understanding, topic-changing

## 🛠️ Development Setup

### Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd emotion-aware-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GOOGLE_API_KEY="your-api-key"
export FLASK_ENV="development"

# Run locally
python cloud_backend.py
```

### Docker Development

```bash
# Build the image
docker build -t emotion-ai .

# Run with environment variables
docker run -p 8080:8080 \
  -e GOOGLE_API_KEY="your-api-key" \
  -e FLASK_ENV="development" \
  emotion-ai
```

### Testing

```bash
# Run the comprehensive test suite
python enhanced_test_script.py

# Test specific functionality
curl -X GET http://localhost:8080/health
```

## 📈 Performance & Monitoring

### Built-in Monitoring

- **Health Checks**: `/health` endpoint with service status
- **Metrics Dashboard**: `/metrics` with performance data
- **Structured Logging**: JSON-formatted logs for analysis
- **Request Tracking**: Response times and error rates

### Performance Optimizations

- **Model Preloading**: Eliminates cold start delays
- **Redis Caching**: Caches similar emotion detections
- **Image Optimization**: Automatic resizing and compression
- **Connection Pooling**: Efficient HTTP connections
- **Rate Limiting**: Prevents resource exhaustion

### Scaling Configuration

```yaml
# Cloud Run auto-scaling settings
- Memory: 2GB (handles ML models)
- CPU: 2 vCPUs (parallel processing)
- Concurrency: 10 (balanced load)
- Max Instances: 5 (cost control)
- Min Instances: 1 (reduced cold starts)
```

## 🔐 Security Features

### Implemented Security

- **Rate Limiting**: API abuse prevention
- **Input Validation**: Sanitization of all inputs
- **Error Handling**: Secure error responses
- **Non-root Containers**: Security hardening
- **HTTPS Enforcement**: Secure communications
- **Secrets Management**: Google Secret Manager integration

### Security Best Practices

```python
# Input validation example
def validate_image_data(image_data):
    if len(img_bytes) > MAX_IMAGE_SIZE:
        return False, "Image too large"
    # Additional validation...

# Rate limiting example
@limiter.limit("10 per minute")
def chat():
    # Protected endpoint
```

## 🎯 Client Features

### Interactive Commands

- `help` - Show available commands
- `status` - Display connection and system status
- `history` - Show recent conversation history
- `save` - Export conversation to JSON file
- `clear` - Clear terminal screen
- `quit/exit/q` - Exit application

### Visual Feedback

The OpenCV window displays:
- **Current emotion** with confidence level
- **Connection status** (connected/disconnected)
- **Frame information** and processing stats
- **Color-coded confidence** (green/yellow/red)

### Conversation Management

- **Local History**: Automatically saved conversations
- **Context Awareness**: References previous messages
- **Export Capability**: Save conversations as JSON
- **Auto-reconnection**: Handles network interruptions

## 🔧 Configuration

### Environment Variables

```bash
# Required
GOOGLE_API_KEY=your-google-ai-api-key

# Optional
REDIS_URL=redis://localhost:6379
MAX_IMAGE_SIZE=2097152
EMOTION_CONFIDENCE_THRESHOLD=0.7
FLASK_ENV=production
PORT=8080
```

### Client Configuration

```python
# In enhanced_local_client.py
CLOUD_SERVER_URL = "https://your-service-url"
FRAME_SKIP = 15  # Process every 15th frame
CONNECTION_TIMEOUT = 10
CHAT_TIMEOUT = 60
MAX_RETRIES = 3
```

## 📊 Usage Examples

### Basic Interaction

```bash
# Start the client
python enhanced_local_client.py

# The client will show:
You (neutral): Hello!
AI: Hello! I'm here and ready to chat. How can I help you today?

You (happy): I just got a promotion!
AI [with context]: That's absolutely wonderful news! Congratulations on your promotion! 🎉 
I can see you're really happy about it, and you should be! This is such an exciting achievement...
```

### API Usage

```python
import requests

# Detect emotion from image
response = requests.post(
    "https://your-service-url/detect_emotion",
    json={"image": base64_image}
)
emotion_data = response.json()

# Chat with detected emotion
response = requests.post(
    "https://your-service-url/chat/context",
    json={
        "user_input": "I'm feeling great today!",
        "emotion": emotion_data["emotion"],
        "user_id": "user123"
    }
)
chat_response = response.json()
```

## 🚨 Troubleshooting

### Common Issues

**Issue**: "Could not open webcam"
```bash
# Solution: Check camera permissions and availability
ls /dev/video*  # Linux
# Ensure no other applications are using the camera
```

**Issue**: "Connection failed"
```bash
# Solution: Verify service URL and network connection
curl https://your-service-url/health
python enhanced_test_script.py
```

**Issue**: "Rate limit exceeded"
```bash
# Solution: Wait or implement request queuing
# Rate limits: 10 chat requests/minute, 30 emotion detections/minute
```

**Issue**: "Low emotion confidence"
```bash
# Solution: Improve lighting, face positioning
# Adjust EMOTION_CONFIDENCE_THRESHOLD if needed
```

### Debug Commands

```bash
# Check service logs
gcloud logs read --limit=50 "resource.type=cloud_run_revision"

# Test all endpoints
python enhanced_test_script.py

# Monitor system resources
curl https://your-service-url/metrics
```

## 🔮 Roadmap

### Planned Features

- **🎙️ Voice Integration**: Speech-to-text and text-to-speech
- **🌐 WebSocket Support**: Real-time bidirectional communication
- **📱 Mobile Apps**: Native iOS and Android clients
- **👥 Multi-user Support**: Room-based conversations
- **🧠 Advanced Context**: Long-term memory and user profiles
- **📊 Analytics Dashboard**: Usage insights and metrics
- **🌍 Multi-language**: International language support
- **🎨 Custom Personalities**: User-defined AI personalities

### Technical Improvements

- **🔄 Event Streaming**: Apache Kafka integration
- **🗄️ Database**: PostgreSQL for persistent storage
- **🔍 Search**: Conversation search and indexing
- **🤖 Model Updates**: Dynamic model versioning
- **⚖️ Load Balancing**: Advanced traffic distribution

## 🤝 Contributing

### Development Process

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Code Standards

- **Python**: Follow PEP 8 style guidelines
- **Testing**: Add tests for new features
- **Documentation**: Update README and docstrings
- **Security**: Follow security best practices

### Development Tools

```bash
# Code formatting
pip install black isort flake8

# Testing
pip install pytest pytest-flask pytest-cov

# Security scanning
pip install bandit safety
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepFace**: Facial emotion recognition
- **Google Generative AI**: Conversational AI capabilities
- **OpenCV**: Computer vision processing
- **Flask**: Web framework
- **Google Cloud**: Cloud infrastructure

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: your-email@example.com

## 📊 Project Stats

- **Language**: Python
- **Framework**: Flask
- **Deployment**: Google Cloud Run
- **ML Models**: DeepFace, Google Generative AI
- **Status**: Production Ready
- **Version**: 2.0.0

---

**Made with ❤️ and 🤖 by [Your Name]**

*Bringing emotions to AI conversations, one smile at a time.*