"""
Configuration template for the Enhanced Emotion-Aware Chatbot.
Copy this file to config.py and customize as needed.
"""

import os
from typing import Dict, Any

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    TESTING = False
    
    # Google AI Configuration
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    GOOGLE_AI_MODEL = os.getenv('GOOGLE_AI_MODEL', 'gemini-1.5-flash')
    
    # Redis Configuration
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    REDIS_TIMEOUT = int(os.getenv('REDIS_TIMEOUT', '5'))
    CACHE_EXPIRATION = int(os.getenv('CACHE_EXPIRATION', '3600'))  # 1 hour
    
    # Application Settings
    MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '2097152'))  # 2MB
    EMOTION_CONFIDENCE_THRESHOLD = float(os.getenv('EMOTION_CONFIDENCE_THRESHOLD', '0.6'))
    MAX_CONVERSATION_HISTORY = int(os.getenv('MAX_CONVERSATION_HISTORY', '10'))
    
    # Rate Limiting Configuration
    RATE_LIMIT_STORAGE_URL = os.getenv('RATE_LIMIT_STORAGE_URL', 'memory://')
    RATE_LIMITS = {
        'default': '200 per day, 50 per hour',
        'chat': '10 per minute',
        'emotion_detection': '30 per minute'
    }
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Security Settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', '16777216'))  # 16MB
    
    # Performance Settings
    GUNICORN_WORKERS = int(os.getenv('GUNICORN_WORKERS', '2'))
    GUNICORN_THREADS = int(os.getenv('GUNICORN_THREADS', '2'))
    GUNICORN_TIMEOUT = int(os.getenv('GUNICORN_TIMEOUT', '300'))
    
    # Model Configuration
    DEEPFACE_MODELS = {
        'emotion': 'emotion',
        'backend': 'opencv',
        'enforce_detection': False
    }
    
    # Enhanced Emotion Prompts with Configurable Personalities
    EMOTION_PROMPTS = {
        "happy": {
            "base": "You are an enthusiastic and joyful chatbot. The user is happy!",
            "instructions": "Match their positive energy with excitement, encouragement, and celebratory responses. Use upbeat language and be genuinely excited for them.",
            "tone": "enthusiastic",
            "energy_level": "high"
        },
        "sad": {
            "base": "You are a compassionate and gentle chatbot. The user seems sad.",
            "instructions": "Respond with deep empathy, offer comfort, and provide emotional support. Be a good listener and offer hope without dismissing their feelings.",
            "tone": "gentle",
            "energy_level": "low"
        },
        "neutral": {
            "base": "You are a balanced and helpful chatbot. The user's emotion is neutral.",
            "instructions": "Provide clear, informative responses while being friendly and approachable. Keep the conversation engaging but not overwhelming.",
            "tone": "balanced",
            "energy_level": "medium"
        },
        "angry": {
            "base": "You are a calm and understanding chatbot. The user appears angry or frustrated.",
            "instructions": "Respond with patience, validate their feelings, and help de-escalate the situation. Avoid being confrontational and offer constructive solutions.",
            "tone": "calm",
            "energy_level": "low"
        },
        "surprise": {
            "base": "You are an enthusiastic and curious chatbot. The user seems surprised!",
            "instructions": "Share in their amazement and wonder. Ask engaging questions and explore their surprise with genuine interest and excitement.",
            "tone": "curious",
            "energy_level": "high"
        },
        "fear": {
            "base": "You are a reassuring and protective chatbot. The user appears fearful or anxious.",
            "instructions": "Provide comfort, security, and calm reassurance. Help them feel safe and offer practical support to address their concerns.",
            "tone": "reassuring",
            "energy_level": "low"
        },
        "disgust": {
            "base": "You are a tactful and understanding chatbot. The user seems disgusted or uncomfortable.",
            "instructions": "Respond with empathy while being respectful of their feelings. Offer to change topics or provide alternative perspectives gently.",
            "tone": "tactful",
            "energy_level": "low"
        }
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    EMOTION_CONFIDENCE_THRESHOLD = 0.5  # Lower threshold for testing
    RATE_LIMITS = {
        'default': '1000 per day, 200 per hour',
        'chat': '50 per minute',
        'emotion_detection': '100 per minute'
    }

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    REDIS_URL = 'redis://localhost:6379/1'  # Use different DB for testing
    EMOTION_CONFIDENCE_THRESHOLD = 0.3
    RATE_LIMITS = {
        'default': '10000 per day',
        'chat': '1000 per minute',
        'emotion_detection': '1000 per minute'
    }

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'INFO'
    # Security settings for production
    CORS_ORIGINS = ['https://yourdomain.com']
    
    # Enhanced rate limiting for production
    RATE_LIMITS = {
        'default': '100 per day, 30 per hour',
        'chat': '5 per minute',
        'emotion_detection': '20 per minute'
    }

class OptimizedConfig(Config):
    """Optimized configuration for high-performance scenarios"""
    # Increased limits for high-traffic scenarios
    GUNICORN_WORKERS = 4
    GUNICORN_THREADS = 4
    MAX_CONVERSATION_HISTORY = 20
    CACHE_EXPIRATION = 7200  # 2 hours
    
    RATE_LIMITS = {
        'default': '500 per day, 100 per hour',
        'chat': '20 per minute',
        'emotion_detection': '60 per minute'
    }

# Configuration factory
config_map = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'optimized': OptimizedConfig,
    'default': Config
}

def get_config(config_name: str = None) -> Config:
    """Get configuration based on environment"""
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'default')
    
    config_class = config_map.get(config_name.lower(), Config)
    return config_class()

# Validation functions
def validate_config(config: Config) -> Dict[str, Any]:
    """Validate configuration and return validation results"""
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Required settings validation
    if not config.GOOGLE_API_KEY:
        results['valid'] = False
        results['errors'].append('GOOGLE_API_KEY is required')
    
    # Threshold validation
    if not 0 <= config.EMOTION_CONFIDENCE_THRESHOLD <= 1:
        results['valid'] = False
        results['errors'].append('EMOTION_CONFIDENCE_THRESHOLD must be between 0 and 1')
    
    # Size validation
    if config.MAX_IMAGE_SIZE < 1024:  # Minimum 1KB
        results['warnings'].append('MAX_IMAGE_SIZE is very small, may cause issues')
    
    if config.MAX_IMAGE_SIZE > 10 * 1024 * 1024:  # 10MB
        results['warnings'].append('MAX_IMAGE_SIZE is very large, may cause memory issues')
    
    # Performance validation
    if config.GUNICORN_WORKERS > 8:
        results['warnings'].append('High number of workers may cause resource contention')
    
    return results

# Environment-specific configurations
def get_database_url(config: Config) -> str:
    """Get database URL based on environment"""
    # This would be used if you add database support
    if config.DEBUG:
        return 'sqlite:///dev.db'
    else:
        return os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/db')

def get_cache_config(config: Config) -> Dict[str, Any]:
    """Get cache configuration"""
    return {
        'url': config.REDIS_URL,
        'timeout': config.REDIS_TIMEOUT,
        'expiration': config.CACHE_EXPIRATION,
        'key_prefix': f"emotion_bot_{os.getenv('FLASK_ENV', 'default')}_"
    }

# Usage example:
if __name__ == '__main__':
    # Example usage
    config = get_config('development')
    validation = validate_config(config)
    
    print(f"Configuration: {config.__class__.__name__}")
    print(f"Valid: {validation['valid']}")
    
    if validation['errors']:
        print("Errors:", validation['errors'])
    
    if validation['warnings']:
        print("Warnings:", validation['warnings'])