# Enhanced Emotion-Aware Chatbot - Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Google Cloud Platform account
- Google Cloud SDK installed
- Docker installed (for local testing)
- Python 3.11+ (for local development)

### 1. Environment Setup

```bash
# Set your project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 2. Set Environment Variables

Create the required environment variables in Google Cloud:

```bash
# Set your Google AI API key
gcloud secrets create GOOGLE_API_KEY --data-file=- <<< "your-api-key-here"

# Optional: Set Redis URL for caching
gcloud secrets create REDIS_URL --data-file=- <<< "redis://your-redis-instance"
```

### 3. Deploy to Google Cloud Run

```bash
# Clone and navigate to your repository
git clone your-repo-url
cd your-repo-directory

# Deploy using Cloud Build
gcloud builds submit --config cloudbuild.yaml .
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GOOGLE_API_KEY` | Google Generative AI API key | - | Yes |
| `REDIS_URL` | Redis connection URL for caching | `redis://localhost:6379` | No |
| `MAX_IMAGE_SIZE` | Maximum image size in bytes | `2097152` (2MB) | No |
| `EMOTION_CONFIDENCE_THRESHOLD` | Minimum confidence for emotion detection | `0.6` | No |
| `FLASK_ENV` | Flask environment | `production` | No |
| `PORT` | Server port | `8080` | No |

### Cloud Run Configuration

The deployment includes optimized settings:
- **Memory**: 2GB (handles ML models efficiently)
- **CPU**: 2 vCPUs (parallel processing)
- **Concurrency**: 10 (balanced load handling)
- **Max Instances**: 5 (cost control)
- **Min Instances**: 1 (reduced cold starts)
- **Timeout**: 300s (handles ML processing)

## 🧪 Testing Your Deployment

### Automated Testing

Run the comprehensive test suite:

```bash
python enhanced_test_script.py
```

### Manual Testing

1. **Health Check**:
   ```bash
   curl https://your-service-url/health
   ```

2. **Basic Chat**:
   ```bash
   curl -X POST https://your-service-url/chat \
     -H "Content-Type: application/json" \
     -d '{"user_input": "Hello!", "emotion": "happy"}'
   ```

3. **Emotion Detection**:
   ```bash
   # You'll need a base64-encoded image
   curl -X POST https://your-service-url/detect_emotion \
     -H "Content-Type: application/json" \
     -d '{"image": "base64-encoded-image-data"}'
   ```

## 🚀 Client Usage

### Enhanced Local Client

```bash
# Install client dependencies
pip install opencv-python requests

# Update the server URL in local_client.py
CLOUD_SERVER_URL = "https://your-service-url"

# Run the client
python local_client.py
```

### Client Features

- **Real-time emotion detection** from webcam
- **Conversation context** maintained across messages
- **Automatic reconnection** with exponential backoff
- **Local conversation history** with export capability
- **Interactive commands** (help, status, history, save)
- **Visual feedback** in OpenCV window

## 🔍 Monitoring and Debugging

### Built-in Monitoring

1. **Health Endpoint**: `/health`
   - Service status
   - Model readiness
   - Memory usage
   - Uptime information

2. **Metrics Endpoint**: `/metrics`
   - Request count
   - Cache statistics
   - System resources
   - Performance metrics

### Logging

The application provides structured logging:

```bash
# View logs in Google Cloud
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=emotion-ai"

# Follow logs in real-time
gcloud logs tail "resource.type=cloud_run_revision AND resource.labels.service_name=emotion-ai"
```

### Common Issues and Solutions

**Issue**: Cold start delays
- **Solution**: Min instances set to 1, models preloaded during build

**Issue**: Memory errors
- **Solution**: 2GB memory allocation, optimized model loading

**Issue**: Rate limiting
- **Solution**: Built-in rate limiting (10 requests/minute for chat)

**Issue**: Emotion detection accuracy
- **Solution**: Configurable confidence threshold, caching for similar images

## 🔐 Security Considerations

### Implemented Security Features

1. **Rate Limiting**: Prevents API abuse
2. **Input Validation**: Sanitizes all inputs
3. **Error Handling**: Prevents information leakage
4. **Non-root Container**: Runs as unprivileged user
5. **Resource Limits**: Prevents resource exhaustion

### Additional Security Recommendations

1. **API Authentication**: Consider adding JWT tokens for production
2. **HTTPS Only**: Enforce SSL/TLS connections
3. **CORS Configuration**: Restrict origins in production
4. **Secrets Management**: Use Google Secret Manager
5. **Regular Updates**: Keep dependencies updated

## 📊 Performance Optimization

### Caching Strategy

1. **Redis Caching**: Similar images cached for 1 hour
2. **LRU Cache**: In-memory caching for frequent requests
3. **Model Preloading**: Eliminates cold start delays

### Resource Management

1. **Connection Pooling**: Efficient HTTP connections
2. **Memory Optimization**: Garbage collection tuning
3. **CPU Optimization**: Multi-threading for I/O operations

## 🔄 CI/CD Pipeline

### Automated Deployment

The `cloudbuild.yaml` includes:

1. **Multi-stage Build**: Optimized Docker images
2. **Health Checks**: Verify deployment success
3. **Blue-Green Deployment**: Zero-downtime updates (optional)
4. **Build Logging**: Comprehensive build tracking

### Manual Deployment Steps

```bash
# Build and test locally
docker build -t emotion-ai .
docker run -p 8080:8080 -e GOOGLE_API_KEY="your-key" emotion-ai

# Deploy to production
gcloud builds submit --config cloudbuild.yaml .
```

## 📈 Scaling Considerations

### Horizontal Scaling

- **Auto-scaling**: Based on request volume
- **Load Balancing**: Automatic with Cloud Run
- **Geographic Distribution**: Multi-region deployment

### Vertical Scaling

- **Memory**: Increase for larger models
- **CPU**: Scale based on processing needs
- **Concurrency**: Adjust based on response times

## 🆘 Troubleshooting

### Common Error Codes

- **400**: Invalid request format
- **429**: Rate limit exceeded
- **500**: Internal server error
- **503**: Service unavailable

### Debug Commands

```bash
# Check service status
gcloud run services describe emotion-ai --region=us-central1

# View recent logs
gcloud logs read --limit=50 "resource.type=cloud_run_revision"

# Test specific endpoints
python enhanced_test_script.py
```

## 📞 Support

For issues and questions:

1. Check the logs first: `gcloud logs read`
2. Run the test suite: `python enhanced_test_script.py`
3. Review the health endpoint: `/health`
4. Check metrics: `/metrics`

## 🔮 Future Enhancements

Planned improvements:

1. **WebSocket Support**: Real-time bidirectional communication
2. **Voice Integration**: Speech-to-text and text-to-speech
3. **Multi-modal Emotion**: Combine visual, audio, and text analysis
4. **User Profiles**: Persistent user preferences and history
5. **Analytics Dashboard**: Usage metrics and insights
6. **Mobile App**: Native iOS/Android clients

---

**Version**: 2.0.0  
**Last Updated**: 2025  
**Compatibility**: Google Cloud Run, Python 3.11+