import requests
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class AudioUtils:
    def __init__(self, groq_api_key: str = None):
        self.groq_api_key = groq_api_key or GROQ_API_KEY
    
    def get_tts_audio(self, text: str, model_name: str = "playai-tts", voice: str = "Fritz-PlayAI", response_format: str = "wav"):
        """
        Calls the Groq TTS API to synthesize speech from text.
        Returns (audio_bytes, error_message). If successful, error_message is None.
        
        Args:
            text: The text to convert to speech
            model_name: The TTS model to use (default: "playai-tts")
            voice: The voice to use (default: "Fritz-PlayAI")
            response_format: The audio format (default: "wav")
        
        Returns:
            tuple: (audio_bytes, error_message) - audio_bytes if successful, None if error
        """
        if not self.groq_api_key:
            return None, "Groq API key not provided"
        
        data = {
            "model": model_name,
            "input": text,
            "voice": voice,
            "response_format": response_format
        }
        headers = {"Authorization": f"Bearer {self.groq_api_key}"}
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/speech",
                json=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.content, None
            else:
                return None, f"TTS failed: {response.text}"
                
        except Exception as e:
            return None, f"TTS request failed: {str(e)}"

    def transcribe_audio_with_groq(self, audio_to_use, audio_label: str, audio_file=None):
        """
        Calls the Groq Speech-to-Text API and returns the transcribed text or error.
        
        Args:
            audio_to_use: tuple for recorded audio or file-like for uploaded audio
            audio_label: 'Recorded audio' or 'Uploaded audio'
            audio_file: the uploaded file (needed for Uploaded audio)
        
        Returns:
            tuple: (transcribed_text, error_message) - transcribed_text if successful, None if error
        """
        if not self.groq_api_key:
            return None, "Groq API key not provided"
        
        try:
            if audio_label == "Recorded audio":
                files = {"file": audio_to_use}
            else:
                files = {"file": audio_file}
            
            data = {
                "model": "whisper-large-v3-turbo",
                "response_format": "text"
            }
            headers = {"Authorization": f"Bearer {self.groq_api_key}"}
            
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.text, None
            else:
                return None, f"Transcription failed: {response.text}"
                
        except Exception as e:
            return None, f"Transcription request failed: {str(e)}"

# Convenience functions for backward compatibility
def get_tts_audio(text: str, model_name: str = "playai-tts", voice: str = "Fritz-PlayAI", response_format: str = "wav"):
    """
    Convenience function that uses the default AudioUtils instance.
    Calls the Groq TTS API to synthesize speech from text.
    Returns (audio_bytes, error_message). If successful, error_message is None.
    """
    audio_utils = AudioUtils()
    return audio_utils.get_tts_audio(text, model_name, voice, response_format)

def transcribe_audio_with_groq(audio_to_use, audio_label: str, audio_file=None):
    """
    Convenience function that uses the default AudioUtils instance.
    Calls the Groq Speech-to-Text API and returns the transcribed text or error.
    audio_to_use: tuple for recorded audio or file-like for uploaded audio
    audio_label: 'Recorded audio' or 'Uploaded audio'
    audio_file: the uploaded file (needed for Uploaded audio)
    """
    audio_utils = AudioUtils()
    return audio_utils.transcribe_audio_with_groq(audio_to_use, audio_label, audio_file)
