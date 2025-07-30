#!/usr/bin/env python3
"""
Test script for RTC audio handler functionality
"""

import os
import sys
import logging
import time
import numpy as np
import io
import wave
import requests

# Add the current directory to the path so we can import rtc_audio_handler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rtc_audio_handler import RealTimeAudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_groq_api_key():
    """Test if Groq API key is available and working"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY environment variable not set")
        return False
    
    logger.info("Testing Groq API key...")
    
    try:
        headers = {"Authorization": f"Bearer {groq_api_key}"}
        data = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=data,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("✅ Groq API key is valid")
            return True
        else:
            logger.error(f"❌ Groq API test failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing Groq API: {e}")
        return False

def test_audio_transcription():
    """Test audio transcription with a simple sine wave"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY environment variable not set")
        return False
    
    logger.info("Testing audio transcription...")
    
    # Create a simple sine wave audio
    sample_rate = 16000
    duration = 2  # 2 seconds
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Convert to int16
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Create WAV bytes
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    wav_buffer.seek(0)
    wav_bytes = wav_buffer.read()
    
    # Test transcription
    try:
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "model": "whisper-large-v3-turbo",
            "response_format": "text"
        }
        headers = {"Authorization": f"Bearer {groq_api_key}"}
        
        logger.info(f"Sending {len(wav_bytes)} bytes to Groq API...")
        response = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            files=files,
            data=data,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.text
            logger.info(f"✅ Transcription successful: {result}")
            return True
        else:
            logger.error(f"❌ Transcription failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during transcription: {e}")
        return False

def test_audio_processor():
    """Test the RealTimeAudioProcessor class"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY environment variable not set")
        return False
    
    logger.info("Testing RealTimeAudioProcessor...")
    
    # Create audio processor
    processor = RealTimeAudioProcessor(groq_api_key=groq_api_key)
    
    # Test API key
    if processor.test_groq_api():
        logger.info("✅ Audio processor Groq API test passed")
    else:
        logger.error("❌ Audio processor Groq API test failed")
        return False
    
    # Test audio transcription
    if test_audio_transcription():
        logger.info("✅ Audio transcription test passed")
    else:
        logger.error("❌ Audio transcription test failed")
        return False
    
    return True

def main():
    """Run all tests"""
    logger.info("Starting RTC audio handler tests...")
    
    # Test 1: Groq API key
    if not test_groq_api_key():
        logger.error("❌ Groq API key test failed")
        return False
    
    # Test 2: Audio transcription
    if not test_audio_transcription():
        logger.error("❌ Audio transcription test failed")
        return False
    
    # Test 3: Audio processor
    if not test_audio_processor():
        logger.error("❌ Audio processor test failed")
        return False
    
    logger.info("✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 