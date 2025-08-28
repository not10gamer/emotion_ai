import subprocess
import shlex
import requests
import json
import time
import base64
import numpy as np
import cv2
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CLOUD_SERVER_URL = "https://emotion-ai-mug7a7ejoa-uc.a.run.app"
TIMEOUT = 30

class APITester:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def log_test_result(self, test_name, success, details=None, response_time=None):
        """Log test result"""
        result = {
            'test_name': test_name,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'response_time': response_time,
            'details': details
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        time_info = f" ({response_time:.2f}s)" if response_time else ""
        logger.info(f"{status} - {test_name}{time_info}")
        if details:
            logger.info(f"Details: {details}")
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/", timeout=TIMEOUT)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.log_test_result(
                    "Root Endpoint", 
                    True, 
                    f"Version: {data.get('version', 'Unknown')}", 
                    response_time
                )
                return True
            else:
                self.log_test_result(
                    "Root Endpoint", 
                    False, 
                    f"HTTP {response.status_code}", 
                    response_time
                )
                return False
        except Exception as e:
            self.log_test_result("Root Endpoint", False, str(e))
            return False
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/health", timeout=TIMEOUT)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                services = data.get('services', {})
                all_healthy = all(services.values())
                
                self.log_test_result(
                    "Health Check", 
                    all_healthy, 
                    f"Services: {services}", 
                    response_time
                )
                return all_healthy
            else:
                self.log_test_result(
                    "Health Check", 
                    False, 
                    f"HTTP {response.status_code}", 
                    response_time
                )
                return False
        except Exception as e:
            self.log_test_result("Health Check", False, str(e))
            return False
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/metrics", timeout=TIMEOUT)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                uptime = data.get('uptime', 0)
                requests_processed = data.get('requests_processed', 0)
                
                self.log_test_result(
                    "Metrics Endpoint", 
                    True, 
                    f"Uptime: {uptime:.1f}s, Requests: {requests_processed}", 
                    response_time
                )
                return True
            else:
                self.log_test_result(
                    "Metrics Endpoint", 
                    False, 
                    f"HTTP {response.status_code}", 
                    response_time
                )
                return False
        except Exception as e:
            self.log_test_result("Metrics Endpoint", False, str(e))
            return False
    
    def create_test_image(self):
        """Create a test image for emotion detection"""
        # Create a simple test image with face-like features
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        # Add face-like features
        # Eyes
        cv2.circle(img, (75, 75), 10, (0, 0, 0), -1)
        cv2.circle(img, (149, 75), 10, (0, 0, 0), -1)
        
        # Nose
        cv2.circle(img, (112, 112), 5, (100, 100, 100), -1)
        
        # Mouth (happy expression)
        cv2.ellipse(img, (112, 150), (20, 10), 0, 0, 180, (50, 50, 50), 2)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return img_base64
    
    def test_emotion_detection(self):
        """Test emotion detection endpoint"""
        try:
            test_image = self.create_test_image()
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/detect_emotion",
                json={'image': test_image},
                timeout=TIMEOUT
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                emotion = data.get('emotion')
                confidence = data.get('confidence', 0)
                
                success = emotion is not None
                self.log_test_result(
                    "Emotion Detection", 
                    success, 
                    f"Emotion: {emotion}, Confidence: {confidence:.2f}", 
                    response_time
                )
                return success
            else:
                self.log_test_result(
                    "Emotion Detection", 
                    False, 
                    f"HTTP {response.status_code}: {response.text}", 
                    response_time
                )
                return False
        except Exception as e:
            self.log_test_result("Emotion Detection", False, str(e))
            return False
    
    def test_chat_endpoint(self):
        """Test basic chat endpoint"""
        try:
            test_payload = {
                'user_input': 'Hello, how are you?',
                'emotion': 'happy'
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/chat",
                json=test_payload,
                timeout=TIMEOUT
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                bot_response = data.get('response')
                emotion_context = data.get('emotion_context')
                
                success = bot_response is not None and len(bot_response.strip()) > 0
                self.log_test_result(
                    "Chat Endpoint", 
                    success, 
                    f"Response length: {len(bot_response) if bot_response else 0}, Context: {emotion_context}", 
                    response_time
                )
                return success
            else:
                self.log_test_result(
                    "Chat Endpoint", 
                    False, 
                    f"HTTP {response.status_code}: {response.text}", 
                    response_time
                )
                return False
        except Exception as e:
            self.log_test_result("Chat Endpoint", False, str(e))
            return False
    
    def test_chat_context_endpoint(self):
        """Test contextual chat endpoint"""
        try:
            test_payload = {
                'user_input': 'What did I just say?',
                'emotion': 'neutral',
                'user_id': 'test_user_123'
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/chat/context",
                json=test_payload,
                timeout=TIMEOUT
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                bot_response = data.get('response')
                user_id = data.get('user_id')
                
                success = bot_response is not None and user_id == 'test_user_123'
                self.log_test_result(
                    "Context Chat Endpoint", 
                    success, 
                    f"Response length: {len(bot_response) if bot_response else 0}, User ID: {user_id}", 
                    response_time
                )
                return success
            else:
                self.log_test_result(
                    "Context Chat Endpoint", 
                    False, 
                    f"HTTP {response.status_code}: {response.text}", 
                    response_time
                )
                return False
        except Exception as e:
            self.log_test_result("Context Chat Endpoint", False, str(e))
            return False
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        try:
            logger.info("Testing rate limiting (this may take a moment)...")
            
            # Send multiple requests quickly to trigger rate limiting
            success_count = 0
            rate_limited_count = 0
            
            for i in range(15):  # Send 15 requests quickly
                try:
                    response = self.session.post(
                        f"{self.base_url}/chat",
                        json={'user_input': f'Test message {i}', 'emotion': 'neutral'},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        success_count += 1
                    elif response.status_code == 429:
                        rate_limited_count += 1
                    
                    time.sleep(0.1)  # Small delay between requests
                except:
                    pass
            
            # Rate limiting should kick in after 10 requests per minute
            rate_limiting_works = rate_limited_count > 0
            
            self.log_test_result(
                "Rate Limiting", 
                rate_limiting_works, 
                f"Successful: {success_count}, Rate Limited: {rate_limited_count}"
            )
            return rate_limiting_works
            
        except Exception as e:
            self.log_test_result("Rate Limiting", False, str(e))
            return False
    
    def test_error_handling(self):
        """Test error handling with invalid requests"""
        try:
            # Test invalid JSON
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/chat",
                data="invalid json",
                headers={'Content-Type': 'application/json'},
                timeout=TIMEOUT
            )
            response_time = time.time() - start_time
            
            # Should return 400 for invalid JSON
            success = response.status_code == 400
            
            self.log_test_result(
                "Error Handling (Invalid JSON)", 
                success, 
                f"HTTP {response.status_code}", 
                response_time
            )
            
            # Test missing required fields
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/chat",
                json={'emotion': 'happy'},  # Missing user_input
                timeout=TIMEOUT
            )
            response_time = time.time() - start_time
            
            success = response.status_code == 400
            
            self.log_test_result(
                "Error Handling (Missing Fields)", 
                success, 
                f"HTTP {response.status_code}", 
                response_time
            )
            
            return True
            
        except Exception as e:
            self.log_test_result("Error Handling", False, str(e))
            return False
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        logger.info("🚀 Starting comprehensive API tests...")
        logger.info(f"Target URL: {self.base_url}")
        
        tests = [
            ("Basic Connectivity", self.test_root_endpoint),
            ("Health Check", self.test_health_endpoint),
            ("Metrics", self.test_metrics_endpoint),
            ("Emotion Detection", self.test_emotion_detection),
            ("Chat Functionality", self.test_chat_endpoint),
            ("Context Chat", self.test_chat_context_endpoint),
            ("Error Handling", self.test_error_handling),
            ("Rate Limiting", self.test_rate_limiting),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n--- Running: {test_name} ---")
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
        
        # Generate summary report
        self.generate_report(passed, total)
        
        return passed == total
    
    def generate_report(self, passed, total):
        """Generate test report"""
        logger.info("\n" + "="*60)
        logger.info("🧪 TEST SUMMARY REPORT")
        logger.info("="*60)
        
        success_rate = (passed / total) * 100 if total > 0 else 0
        logger.info(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! API is working correctly.")
        elif passed > total * 0.8:
            logger.info("⚠️ Most tests passed, but some issues detected.")
        else:
            logger.info("❌ Multiple test failures detected. Check the logs above.")
        
        logger.info("\nDetailed Results:")
        for result in self.test_results:
            status = "✅" if result['success'] else "❌"
            time_info = f" ({result['response_time']:.2f}s)" if result['response_time'] else ""
            logger.info(f"{status} {result['test_name']}{time_info}")
            if result['details']:
                logger.info(f"   Details: {result['details']}")
        
        logger.info("="*60)

def test_connection_with_curl(server_url):
    """Test connection using curl as fallback"""
    logger.info(f"🔧 Testing connection with curl...")
    logger.info(f"Target: {server_url}")
    
    command = f"curl -X POST -H \"Content-Type: application/json\" -d '{{\"user_input\": \"hello\", \"emotion\": \"neutral\"}}' {server_url}/chat"
    logger.info(f"Command: {command}")
    
    try:
        process = subprocess.Popen(
            shlex.split(command), 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        stdout, stderr = process.communicate(timeout=60)
        
        logger.info("\n--- CURL Response ---")
        if process.returncode == 0:
            logger.info("✅ Connection successful!")
            logger.info("Response from server:")
            try:
                response_data = json.loads(stdout)
                logger.info(json.dumps(response_data, indent=2))
            except:
                logger.info(stdout)
        else:
            logger.info("❌ Connection failed.")
            logger.info(f"Return Code: {process.returncode}")
            logger.info(f"STDOUT: {stdout}")
            logger.info(f"STDERR: {stderr}")
    
    except subprocess.TimeoutExpired:
        logger.error("⏰ Connection timed out. The server might be overloaded.")
    except Exception as e:
        logger.error(f"💥 Unexpected error occurred: {e}")

def main():
    """Main test function"""
    logger.info("🎯 Enhanced API Testing Suite")
    logger.info("="*60)
    
    # Run comprehensive API tests
    tester = APITester(CLOUD_SERVER_URL)
    all_passed = tester.run_all_tests()
    
    # Run curl test as backup
    logger.info("\n" + "="*60)
    logger.info("🔧 Running curl backup test...")
    test_connection_with_curl(CLOUD_SERVER_URL)
    
    # Final summary
    logger.info("\n" + "="*60)
    if all_passed:
        logger.info("🎉 All tests completed successfully!")
        logger.info("Your API is ready for production use.")
    else:
        logger.info("⚠️ Some tests failed. Please review the results above.")
        logger.info("Consider investigating the failed endpoints before deploying.")
    
    logger.info("="*60)
    
    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Testing interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"💥 Fatal error during testing: {e}")
        exit(1)