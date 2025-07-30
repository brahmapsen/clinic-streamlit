import asyncio
import logging
import queue
import threading
import time
import uuid
from typing import Optional, Callable
import numpy as np
import av
import requests
import os
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from aiortc.contrib.media import MediaRecorder
import streamlit as st

# Configure logging to be more verbose for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeAudioProcessor:
    def __init__(self, groq_api_key: str, on_transcription: Callable[[str], None] = None):
        self.groq_api_key = groq_api_key
        self.on_transcription = on_transcription
        self.audio_buffer = queue.Queue()
        self.is_recording = False
        self.transcription_thread = None
        self.audio_frames = []
        self.sample_rate = 16000
        self.chunk_duration = 3.0  # Process audio in 3-second chunks
        self.silence_threshold = 0.01  # Audio level below this is considered silence (adjusted for normalized levels)
        self.silence_duration = 5.0  # Seconds of silence before auto-transcription
        self.last_audio_time = 0  # Track when we last heard audio
        self.silence_start_time = None  # When silence started
        self.frame_count = 0
        self.last_audio_level = 0
        self.worker_running = False
        
        # Volume-based silence detection parameters
        self.volume_history = []  # Store recent volume levels for adaptive threshold
        self.max_history_size = 100  # Keep last 100 volume measurements
        self.adaptive_threshold_multiplier = 0.3  # Threshold as fraction of recent average
        self.min_silence_duration = 1.5  # Minimum silence duration (seconds)
        self.max_silence_duration = 8.0  # Maximum silence duration (seconds)
        self.volume_peak_threshold = 0.05  # Minimum volume to be considered speech
        self.consecutive_silence_frames = 0  # Count consecutive silent frames
        self.min_speech_frames = 10  # Minimum frames with speech before considering silence
        self.speech_detected = False  # Track if we've detected meaningful speech
        
        logger.info(f"RealTimeAudioProcessor initialized with Groq API key: {'*' * 10 if groq_api_key else 'None'}")
        
    def audio_frame_callback(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Process incoming audio frames from WebRTC"""
        try:
            # Always process frames for debugging, but only record when is_recording is True
            # Convert audio frame to numpy array - handle different formats
            try:
                audio_array = frame.to_ndarray()
                
                # Handle multi-dimensional arrays (stereo, multi-channel)
                if audio_array.ndim > 1:
                    # Convert to mono by averaging channels
                    audio_array = np.mean(audio_array, axis=0)
                
                # Ensure it's a 1D array
                audio_array = audio_array.flatten()
                
                # Convert to float32 for processing
                if audio_array.dtype != np.float32:
                    if audio_array.dtype == np.int16:
                        audio_array = audio_array.astype(np.float32) / 32768.0
                    elif audio_array.dtype == np.int32:
                        audio_array = audio_array.astype(np.float32) / 2147483648.0
                    else:
                        audio_array = audio_array.astype(np.float32)
                
            except Exception as e:
                logger.error(f"Error converting audio frame to array: {e}")
                return frame
            
            # Debug: Store frame info for status display
            self.frame_count += 1
            # Calculate audio level properly based on data type
            if audio_array.dtype == np.int16:
                # For int16, normalize to 0-1 range for level calculation
                audio_level = np.abs(audio_array.astype(np.float32) / 32768.0).mean()
            else:
                # For float32, use as is
                audio_level = np.abs(audio_array).mean()
            
            self.last_audio_level = audio_level
            
            # Log every 100th frame to avoid spam
            if self.frame_count % 100 == 0:
                logger.info(f"Processed {self.frame_count} frames, current audio level: {self.last_audio_level:.4f}")
            
            # Only process audio when recording is active
            if self.is_recording:
                # Volume-based silence detection
                current_time = time.time()
                
                # Update volume history for adaptive threshold
                self.volume_history.append(self.last_audio_level)
                if len(self.volume_history) > self.max_history_size:
                    self.volume_history.pop(0)
                
                # Calculate adaptive threshold based on recent volume history
                if len(self.volume_history) >= 20:  # Need some history
                    recent_avg = np.mean(self.volume_history[-20:])  # Last 20 measurements
                    adaptive_threshold = recent_avg * self.adaptive_threshold_multiplier
                    # Ensure minimum threshold
                    adaptive_threshold = max(adaptive_threshold, 0.005)
                else:
                    adaptive_threshold = self.silence_threshold
                
                # Detect speech vs silence
                is_speech = self.last_audio_level > adaptive_threshold
                
                if is_speech:
                    # Speech detected
                    self.consecutive_silence_frames = 0
                    self.last_audio_time = current_time
                    
                    # Check if this is meaningful speech (above peak threshold)
                    if self.last_audio_level > self.volume_peak_threshold:
                        if not self.speech_detected:
                            logger.info(f"Speech started (level: {self.last_audio_level:.4f}, threshold: {adaptive_threshold:.4f})")
                        self.speech_detected = True
                    
                    # Reset silence tracking
                    if self.silence_start_time is not None:
                        silence_duration = current_time - self.silence_start_time
                        logger.info(f"Speech detected after {silence_duration:.1f}s of silence (level: {self.last_audio_level:.4f})")
                    self.silence_start_time = None
                    
                else:
                    # Silence detected
                    self.consecutive_silence_frames += 1
                    
                    # Only start silence tracking if we've detected meaningful speech
                    if self.speech_detected and self.silence_start_time is None:
                        self.silence_start_time = current_time
                        logger.info(f"Silence started after speech (level: {self.last_audio_level:.4f}, threshold: {adaptive_threshold:.4f})")
                    
                    # Check if we should trigger transcription
                    if (self.silence_start_time is not None and 
                        self.speech_detected and 
                        len(self.audio_frames) > 0):
                        
                        silence_duration = current_time - self.silence_start_time
                        
                        # Check if silence duration is within acceptable range
                        if (self.min_silence_duration <= silence_duration <= self.max_silence_duration and
                            self.consecutive_silence_frames >= 50):  # At least 50 consecutive silent frames
                            
                            logger.info(f"Volume-based silence detected - duration: {silence_duration:.1f}s, frames: {self.consecutive_silence_frames}")
                            
                            # Check if we have enough audio content
                            min_audio_length = int(self.sample_rate * 0.5)  # 0.5 second minimum
                            if len(self.audio_frames) >= min_audio_length:
                                # Check if audio has sufficient energy
                                audio_array_float = np.array(self.audio_frames, dtype=np.float32)
                                audio_energy = np.mean(np.abs(audio_array_float))
                                
                                logger.info(f"Audio buffer: {len(self.audio_frames)} samples, energy: {audio_energy:.4f}")
                                
                                if audio_energy > adaptive_threshold:
                                    logger.info(f"Triggering auto-transcription (energy: {audio_energy:.4f})")
                                    # Convert to int16 for WAV processing
                                    chunk = (np.array(self.audio_frames, dtype=np.float32) * 32767).astype(np.int16)
                                    self.audio_frames = []  # Clear the buffer
                                    self.audio_buffer.put(chunk)
                                    logger.info(f"Audio chunk added to buffer for transcription - chunk size: {len(chunk)}")
                                    
                                    # Reset for next speech segment
                                    self.speech_detected = False
                                    self.silence_start_time = None
                                    self.consecutive_silence_frames = 0
                                else:
                                    logger.info(f"Audio energy too low ({audio_energy:.4f}) - skipping transcription")
                                    self.audio_frames = []
                            else:
                                logger.info(f"Audio too short ({len(self.audio_frames)} samples) - skipping transcription")
                                self.audio_frames = []
                
                # Resample to 16kHz if needed (Groq expects 16kHz)
                if frame.sample_rate != self.sample_rate:
                    # Simple resampling (for production, use proper resampling)
                    ratio = self.sample_rate / frame.sample_rate
                    audio_array = np.interp(
                        np.linspace(0, len(audio_array), int(len(audio_array) * ratio)),
                        np.arange(len(audio_array)),
                        audio_array
                    ).astype(np.float32)
                
                # Store as float32 for processing
                self.audio_frames.extend(audio_array.flatten().tolist())
                logger.debug(f"Added {len(audio_array)} samples to audio_frames (total: {len(self.audio_frames)})")
        
        except Exception as e:
            logger.error(f"Error in audio_frame_callback: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return frame
    
    def start_recording(self):
        """Start recording and transcription"""
        logger.info("Starting recording...")
        self.is_recording = True
        self.audio_frames = []
        self.silence_start_time = None
        self.last_audio_time = time.time()
        
        # Start transcription thread
        if self.transcription_thread is None or not self.transcription_thread.is_alive():
            self.worker_running = True
            self.transcription_thread = threading.Thread(target=self._transcription_worker)
            self.transcription_thread.daemon = True
            self.transcription_thread.start()
            logger.info("Transcription worker started")
    
    def stop_recording(self):
        """Stop recording and process remaining audio"""
        logger.info("Stopping recording...")
        self.is_recording = False
        
        # Process any remaining audio
        if len(self.audio_frames) > 0:
            logger.info(f"Processing remaining {len(self.audio_frames)} audio samples")
            # Convert float32 to int16 for WAV processing
            chunk = (np.array(self.audio_frames, dtype=np.float32) * 32767).astype(np.int16)
            self.audio_buffer.put(chunk)
            self.audio_frames = []
    
    def _transcription_worker(self):
        """Background worker for processing audio chunks"""
        logger.info("Transcription worker started")
        while self.worker_running:
            try:
                # Get audio chunk from queue (with timeout)
                try:
                    audio_chunk = self.audio_buffer.get(timeout=1.0)
                    logger.info(f"Transcription worker received audio chunk of size {len(audio_chunk)}")
                except queue.Empty:
                    if not self.worker_running:
                        logger.info("Worker stopping - no more audio chunks")
                        break
                    continue
                
                # Check if audio chunk has sufficient content before transcribing
                # Convert back to float32 range for energy calculation
                audio_chunk_float = audio_chunk.astype(np.float32) / 32767.0
                audio_energy = np.mean(np.abs(audio_chunk_float))
                # Use a lower threshold for transcription since we already filtered in the main loop
                min_energy_threshold = 0.005  # Lower threshold for transcription
                
                logger.info(f"Audio chunk energy: {audio_energy:.4f}, threshold: {min_energy_threshold:.4f}")
                
                if audio_energy < min_energy_threshold:
                    logger.info(f"Audio energy too low ({audio_energy:.4f}) - skipping transcription")
                    self.audio_buffer.task_done()
                    continue
                
                # Convert numpy array to WAV bytes
                wav_bytes = self._numpy_to_wav(audio_chunk)
                
                # Debug: Log transcription attempt
                logger.info(f"Attempting transcription of {len(wav_bytes)} bytes (energy: {audio_energy:.4f})")
                
                # Transcribe with Groq
                transcription = self._transcribe_with_groq(wav_bytes)
                
                # Debug: Log transcription result
                logger.info(f"Transcription result: {transcription}")
                
                # Only process meaningful transcriptions
                if transcription and transcription.strip() and self.on_transcription:
                    cleaned_transcription = transcription.strip()
                    
                    # Filter out very short or noise-like transcriptions
                    if len(cleaned_transcription) >= 3:  # Minimum meaningful length
                        logger.info(f"Calling on_transcription with: {cleaned_transcription}")
                        self.on_transcription(cleaned_transcription)
                    else:
                        logger.info(f"Transcription too short ({len(cleaned_transcription)} chars) - skipping")
                else:
                    logger.warning(f"No meaningful transcription. Text: {transcription}, Callback: {self.on_transcription}")
                
                self.audio_buffer.task_done()
                
            except Exception as e:
                logger.error(f"Error in transcription worker: {e}")
                import traceback
                logger.error(traceback.format_exc())
                try:
                    self.audio_buffer.task_done()
                except:
                    pass
        
        logger.info("Transcription worker stopped")
    
    def _numpy_to_wav(self, audio_data: np.ndarray) -> bytes:
        """Convert numpy array to WAV bytes"""
        import io
        import wave
        
        # Ensure audio is int16
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    def _transcribe_with_groq(self, wav_bytes: bytes) -> Optional[str]:
        """Transcribe audio using Groq API"""
        try:
            files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
            data = {
                "model": "whisper-large-v3-turbo",
                "response_format": "text"
            }
            headers = {"Authorization": f"Bearer {self.groq_api_key}"}
            
            logger.info(f"Sending {len(wav_bytes)} bytes to Groq API")
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.text
                logger.info(f"Groq transcription successful: {result}")
                return result
            else:
                logger.error(f"Groq transcription failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None
    
    def test_groq_api(self):
        """Test if Groq API is accessible"""
        try:
            # Test with a simple text completion to verify API key
            headers = {"Authorization": f"Bearer {self.groq_api_key}"}
            data = {
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=data,
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info("Groq API test successful - API key is valid")
                return True
            else:
                logger.error(f"Groq API test failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing Groq API: {e}")
            return False

def create_webrtc_audio_streamer(
    key: str,
    audio_processor: RealTimeAudioProcessor,
    is_active: bool = False
) -> Optional[object]:
    """Create WebRTC audio streamer component"""
    
    webrtc_ctx = webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDONLY,
        audio_frame_callback=audio_processor.audio_frame_callback,
        media_stream_constraints={
            "video": False,
            "audio": {
                "deviceId": "default",  # Use default built-in audio
                "sampleRate": 16000,
                "channelCount": 1,
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True,
            }
        },
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )
    
    return webrtc_ctx

class RealTimeVoiceInterface:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.audio_processor = None
        self.webrtc_ctx = None
        self.transcriptions = []
        self.transcription_queue = queue.Queue()  # Queue for transcriptions from background thread
        
        # Initialize session state
        if 'rtc_transcriptions' not in st.session_state:
            st.session_state.rtc_transcriptions = []
        if 'rtc_is_recording' not in st.session_state:
            st.session_state.rtc_is_recording = False
            
        logger.info(f"RealTimeVoiceInterface initialized with Groq API key: {'*' * 10 if groq_api_key else 'None'}")
    
    def on_transcription_received(self, transcription: str):
        """Callback when transcription is received - put in queue for main thread processing"""
        logger.info(f"on_transcription_received called with: {transcription}")
        
        try:
            # Put transcription in queue for main thread to process
            self.transcription_queue.put({
                'text': transcription,
                'timestamp': time.time()
            })
            logger.info(f"Added transcription to queue: {transcription}")
            
        except Exception as e:
            logger.error(f"Error in on_transcription_received: {e}")
            # Store transcription in a temporary variable if queue fails
            if not hasattr(self, 'temp_transcriptions'):
                self.temp_transcriptions = []
            self.temp_transcriptions.append(transcription)
    
    def _process_transcription_queue(self):
        """Process transcriptions from the queue in the main thread"""
        new_transcriptions = []
        
        # Get all available transcriptions from queue
        while not self.transcription_queue.empty():
            try:
                transcription_data = self.transcription_queue.get_nowait()
                new_transcriptions.append(transcription_data)
                logger.info(f"Processing queued transcription: {transcription_data['text']}")
            except queue.Empty:
                break
        
        # Process new transcriptions
        if new_transcriptions:
            # Add to session state
            for trans_data in new_transcriptions:
                st.session_state.rtc_transcriptions.append(trans_data)
            
            # Update user input with combined transcriptions
            combined_text = " ".join([t['text'] for t in st.session_state.rtc_transcriptions])
            st.session_state['user_input'] = combined_text
            
            logger.info(f"Updated user_input to: {combined_text}")
            
            # Automatically send to backend
            self._auto_send_message(combined_text)
            
            return True  # Indicate that new transcriptions were processed
        
        return False  # No new transcriptions
    
    def _auto_send_message(self, message: str):
        """Automatically send transcribed message to backend - only if meaningful content"""
        # Check if message has meaningful content
        if not message or not message.strip():
            logger.info("No message content - skipping auto-send")
            return
            
        # Filter out very short or meaningless messages
        cleaned_message = message.strip()
        if len(cleaned_message) < 3:  # Too short to be meaningful
            logger.info(f"Message too short ({len(cleaned_message)} chars) - skipping auto-send")
            return
            
        # Check for common noise words/sounds that shouldn't trigger sending
        noise_patterns = ['um', 'uh', 'hmm', 'ah', 'er', 'oh']
        if cleaned_message.lower() in noise_patterns:
            logger.info(f"Message appears to be noise ({cleaned_message}) - skipping auto-send")
            return
            
        try:
            import requests
            
            # Safely access session state
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            if 'user_id' not in st.session_state:
                st.session_state.user_id = f"user_{str(uuid.uuid4())[:8]}"
            if 'api_url' not in st.session_state:
                st.session_state.api_url = "http://localhost:8000"  # Default API URL
            
            # Add user message to chat history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': cleaned_message
            })
            
            # Prepare API request with conversation context
            payload = {
                "user_id": st.session_state.user_id,
                "message": cleaned_message,
                "language": "en",
                "conversation_history": st.session_state.chat_history
            }
            
            logger.info(f"Auto-sending message to backend: {cleaned_message}")
            
            # Send to backend
            response = requests.post(
                f"{st.session_state.api_url}/triage",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    assistant_response = response_data.get("response", "No response received")
                    
                    # Add assistant response to chat
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': assistant_response
                    })
                    
                    logger.info(f"Received response from backend: {assistant_response}")
                    
                    # Clear the transcription stream after successful processing
                    self._clear_audio_stream()
                    
                except Exception as e:
                    logger.error(f"Error parsing backend response: {e}")
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': f"Error processing response: {str(e)}"
                    })
            else:
                logger.error(f"Backend API error: {response.status_code} - {response.text}")
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': f"API Error: {response.status_code}"
                })
                
        except Exception as e:
            logger.error(f"Error auto-sending message: {e}")
            # Only try to append to chat history if session state is accessible
            try:
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': f"Connection error: {str(e)}"
                })
            except:
                logger.error("Could not access session state to log error")
    
    def _clear_audio_stream(self):
        """Clear the audio stream and reset transcription state"""
        try:
            # Safely clear transcriptions from session state
            try:
                if 'rtc_transcriptions' in st.session_state:
                    st.session_state.rtc_transcriptions = []
                if 'user_input' in st.session_state:
                    st.session_state['user_input'] = ""
                # Set flag to show stream cleared message
                st.session_state.stream_cleared = True
            except Exception as session_error:
                logger.warning(f"Could not access session state for clearing: {session_error}")
            
            # Reset audio processor state if available
            if self.audio_processor:
                self.audio_processor.audio_frames = []
                # Clear any remaining items in the audio buffer
                while not self.audio_processor.audio_buffer.empty():
                    try:
                        self.audio_processor.audio_buffer.get_nowait()
                    except:
                        break
                        
                # Reset silence tracking
                self.audio_processor.silence_start_time = None
                self.audio_processor.last_audio_time = 0
                
            logger.info("Audio stream cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing audio stream: {e}")
    
    def render_interface(self):
        """Render the real-time voice interface"""
        
        # Process any pending transcriptions from the queue
        new_transcriptions_processed = self._process_transcription_queue()
        
        # If new transcriptions were processed, trigger a rerun to update the UI
        if new_transcriptions_processed:
            st.rerun()
        
        # Initialize audio processor
        if self.audio_processor is None:
            self.audio_processor = RealTimeAudioProcessor(
                groq_api_key=self.groq_api_key,
                on_transcription=self.on_transcription_received
            )
            
            # Test Groq API key
            if self.groq_api_key:
                api_test_result = self.audio_processor.test_groq_api()
                if api_test_result:
                    st.success("✅ Groq API key is valid")
                else:
                    st.error("❌ Groq API key test failed - check your API key")
        
        # Debug section
        with st.expander("🔧 Debug Options"):
            if st.button("Test Groq API"):
                if self.audio_processor:
                    api_test_result = self.audio_processor.test_groq_api()
                    if api_test_result:
                        st.success("✅ Groq API test successful")
                    else:
                        st.error("❌ Groq API test failed")
                else:
                    st.error("Audio processor not initialized")
            
            if st.button("Show Audio Debug Info"):
                if self.audio_processor:
                    st.write(f"Recording: {self.audio_processor.is_recording}")
                    st.write(f"Frames processed: {self.audio_processor.frame_count}")
                    st.write(f"Audio level: {self.audio_processor.last_audio_level:.4f}")
                    st.write(f"Audio frames buffer size: {len(self.audio_processor.audio_frames)}")
                    st.write(f"Queue size: {self.audio_processor.audio_buffer.qsize()}")
                    st.write(f"Worker running: {self.audio_processor.worker_running}")
                    if self.audio_processor.transcription_thread:
                        st.write(f"Worker alive: {self.audio_processor.transcription_thread.is_alive()}")
                    
                    # Volume-based detection info
                    if hasattr(self.audio_processor, 'volume_history'):
                        st.write(f"Volume history size: {len(self.audio_processor.volume_history)}")
                        if len(self.audio_processor.volume_history) >= 20:
                            recent_avg = np.mean(self.audio_processor.volume_history[-20:])
                            adaptive_threshold = recent_avg * self.audio_processor.adaptive_threshold_multiplier
                            adaptive_threshold = max(adaptive_threshold, 0.005)
                            st.write(f"Recent volume average: {recent_avg:.4f}")
                            st.write(f"Adaptive threshold: {adaptive_threshold:.4f}")
                            st.write(f"Speech detected: {self.audio_processor.speech_detected}")
                            st.write(f"Consecutive silence frames: {self.audio_processor.consecutive_silence_frames}")
                else:
                    st.error("Audio processor not initialized")
            
            if st.button("Test Auto-Transcription"):
                if self.audio_processor and self.groq_api_key:
                    # Generate a simple test audio (sine wave)
                    import numpy as np
                    import io
                    import wave
                    
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
                    
                    st.info("Testing transcription with generated audio...")
                    transcription = self.audio_processor._transcribe_with_groq(wav_bytes)
                    if transcription:
                        st.success(f"Test transcription successful: {transcription}")
                    else:
                        st.error("Test transcription failed")
                else:
                    st.error("Audio processor or API key not available")
            
            if st.button("Test Silence Detection"):
                if self.audio_processor and self.audio_processor.is_recording:
                    # Simulate the silence detection logic
                    if len(self.audio_processor.audio_frames) > 0:
                        min_audio_length = int(self.audio_processor.sample_rate * 1.0)
                        if len(self.audio_processor.audio_frames) >= min_audio_length:
                            audio_array_float = np.array(self.audio_processor.audio_frames, dtype=np.float32)
                            audio_energy = np.mean(np.abs(audio_array_float))
                            
                            st.info(f"Audio frames: {len(self.audio_processor.audio_frames)}")
                            st.info(f"Audio energy: {audio_energy:.4f}")
                            st.info(f"Silence threshold: {self.audio_processor.silence_threshold}")
                            
                            if audio_energy > self.audio_processor.silence_threshold:
                                st.success("✅ Audio has sufficient energy for transcription")
                                # Manually trigger transcription
                                chunk = (np.array(self.audio_processor.audio_frames, dtype=np.float32) * 32767).astype(np.int16)
                                self.audio_processor.audio_buffer.put(chunk)
                                self.audio_processor.audio_frames = []
                                st.success("✅ Manually triggered transcription")
                            else:
                                st.warning("⚠️ Audio energy too low for transcription")
                        else:
                            st.warning(f"⚠️ Audio too short ({len(self.audio_processor.audio_frames)} samples)")
                    else:
                        st.warning("⚠️ No audio frames captured yet")
                else:
                    st.error("Audio processor not recording")
            
            if st.button("Force Transcription Now"):
                if self.audio_processor and self.audio_processor.is_recording:
                    if len(self.audio_processor.audio_frames) > 0:
                        st.info(f"Forcing transcription with {len(self.audio_processor.audio_frames)} audio samples...")
                        # Force transcription regardless of energy
                        chunk = (np.array(self.audio_processor.audio_frames, dtype=np.float32) * 32767).astype(np.int16)
                        self.audio_processor.audio_buffer.put(chunk)
                        self.audio_processor.audio_frames = []
                        st.success("✅ Forced transcription triggered")
                    else:
                        st.warning("⚠️ No audio frames to transcribe")
                else:
                    st.error("Audio processor not recording")
            
            if st.button("Test Volume-Based Detection"):
                if self.audio_processor and self.audio_processor.is_recording:
                    if len(self.audio_processor.volume_history) >= 20:
                        recent_avg = np.mean(self.audio_processor.volume_history[-20:])
                        adaptive_threshold = recent_avg * self.audio_processor.adaptive_threshold_multiplier
                        adaptive_threshold = max(adaptive_threshold, 0.005)
                        
                        st.info(f"Current audio level: {self.audio_processor.last_audio_level:.4f}")
                        st.info(f"Adaptive threshold: {adaptive_threshold:.4f}")
                        st.info(f"Speech detected: {self.audio_processor.speech_detected}")
                        st.info(f"Consecutive silence frames: {self.audio_processor.consecutive_silence_frames}")
                        
                        if self.audio_processor.last_audio_level > adaptive_threshold:
                            st.success("✅ Currently detecting speech")
                        else:
                            st.warning("🔇 Currently detecting silence")
                    else:
                        st.warning("⚠️ Need more volume history for adaptive threshold")
                else:
                    st.error("Audio processor not recording")
            
            if st.button("Test Transcription Pipeline"):
                if self.audio_processor and self.groq_api_key:
                    # Create a simple test audio with speech-like content
                    import numpy as np
                    import io
                    import wave
                    
                    # Create a simple sine wave that sounds like "hello"
                    sample_rate = 16000
                    duration = 2  # 2 seconds
                    t = np.linspace(0, duration, int(sample_rate * duration))
                    
                    # Create a more complex waveform that might be transcribed
                    audio_data = (np.sin(2 * np.pi * 440 * t) * 0.1 +  # A4
                                np.sin(2 * np.pi * 880 * t) * 0.05 +   # A5
                                np.sin(2 * np.pi * 220 * t) * 0.05)    # A3
                    
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
                    
                    st.info("Testing transcription pipeline...")
                    transcription = self.audio_processor._transcribe_with_groq(wav_bytes)
                    if transcription:
                        st.success(f"✅ Test transcription successful: {transcription}")
                    else:
                        st.error("❌ Test transcription failed")
                else:
                    st.error("Audio processor or API key not available")
        
        # Control buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.rtc_is_recording:
                if st.button("⏹️ Stop Recording"):
                    st.session_state.rtc_is_recording = False
                    if self.audio_processor:
                        self.audio_processor.stop_recording()
                    st.rerun()
            else:
                if st.button("🎤 Start Recording"):
                    st.session_state.rtc_is_recording = True
                    if self.audio_processor:
                        self.audio_processor.start_recording()
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All"):
                st.session_state.rtc_transcriptions = []
                st.session_state['user_input'] = ""
                st.rerun()
        
        # WebRTC streamer - only create when recording is active
        if st.session_state.rtc_is_recording:
            self.webrtc_ctx = create_webrtc_audio_streamer(
                key="realtime-audio",
                audio_processor=self.audio_processor,
                is_active=True
            )
        else:
            # Show placeholder when not recording
            st.info("🎤 Click 'Start Recording' to begin voice input")
            self.webrtc_ctx = None
        
        # Show WebRTC status
        if self.webrtc_ctx:
            if self.webrtc_ctx.state.playing:
                st.success("🟢 Microphone connected and ready!")
                
                # Show audio processing debug info
                if self.audio_processor and hasattr(self.audio_processor, 'frame_count'):
                    col_debug1, col_debug2, col_debug3 = st.columns(3)
                    with col_debug1:
                        st.metric("Frames Processed", self.audio_processor.frame_count)
                    with col_debug2:
                        audio_level = getattr(self.audio_processor, 'last_audio_level', 0)
                        st.metric("Audio Level", f"{audio_level:.4f}")
                    with col_debug3:
                        queue_size = self.audio_processor.audio_buffer.qsize() if self.audio_processor else 0
                        st.metric("Queue Size", queue_size)
                    
                    # Show audio detection and silence tracking
                    if audio_level > 0.001:
                        st.info("🎤 Audio detected! Speaking...")
                    elif st.session_state.rtc_is_recording:
                        # Show silence countdown
                        if hasattr(self.audio_processor, 'silence_start_time') and self.audio_processor.silence_start_time:
                            silence_duration = time.time() - self.audio_processor.silence_start_time
                            remaining = max(0, 5.0 - silence_duration)
                            if remaining > 0:
                                st.warning(f"🔇 Silence detected. Auto-transcription in {remaining:.1f}s...")
                            else:
                                st.info("⏳ Processing audio for transcription...")
                        else:
                            st.warning("🔇 No audio detected. Try speaking louder.")
                    
                    # Show transcription status
                    if hasattr(self.audio_processor, 'transcription_thread') and self.audio_processor.transcription_thread:
                        if self.audio_processor.transcription_thread.is_alive():
                            st.info("🔄 Transcription worker is running")
                        else:
                            st.warning("⚠️ Transcription worker stopped")
                    
                    # Show queue status
                    queue_size = self.audio_processor.audio_buffer.qsize() if self.audio_processor else 0
                    if queue_size > 0:
                        st.warning(f"⏳ {queue_size} audio chunks waiting for transcription")
                    else:
                        st.info("✅ No audio chunks in queue")
                    
                    # Show if we have a valid Groq API key
                    if self.groq_api_key and len(self.groq_api_key) > 10:
                        st.success("🔑 Groq API key configured")
                    else:
                        st.error("❌ Groq API key missing or invalid")
                    
                    # Show audio frame processing status
                    if self.audio_processor.frame_count > 0:
                        st.success(f"🎵 Audio frames being received ({self.audio_processor.frame_count} processed)")
                    else:
                        st.warning("⚠️ No audio frames received yet")
                    
                    # Show audio buffer status
                    if self.audio_processor.is_recording:
                        buffer_size = len(self.audio_processor.audio_frames)
                        if buffer_size > 0:
                            st.info(f"📊 Audio buffer: {buffer_size} samples ({buffer_size/16000:.1f}s)")
                        else:
                            st.warning("📊 Audio buffer: Empty")
                        
                        # Show volume-based silence status
                        if hasattr(self.audio_processor, 'volume_history') and len(self.audio_processor.volume_history) >= 20:
                            import numpy as np
                            recent_avg = np.mean(self.audio_processor.volume_history[-20:])
                            adaptive_threshold = recent_avg * self.audio_processor.adaptive_threshold_multiplier
                            adaptive_threshold = max(adaptive_threshold, 0.005)
                            
                            st.info(f"🎵 Current level: {self.audio_processor.last_audio_level:.4f}")
                            st.info(f"📏 Adaptive threshold: {adaptive_threshold:.4f}")
                            
                            if self.audio_processor.speech_detected:
                                if self.audio_processor.silence_start_time is not None:
                                    silence_duration = time.time() - self.audio_processor.silence_start_time
                                    st.warning(f"🔇 Silence for {silence_duration:.1f}s ({self.audio_processor.consecutive_silence_frames} frames)")
                                else:
                                    st.success("🎤 Speaking detected")
                            else:
                                st.info("⏳ Waiting for speech...")
                        else:
                            st.info("🔄 Calibrating volume thresholds...")
        
        # Recording status with custom styling
        if st.session_state.rtc_is_recording:
            st.markdown("""
            <div class="rtc-recording-active">
                🔴 RECORDING... Speak now! Your voice is being transcribed in real-time.
            </div>
            """, unsafe_allow_html=True)
            
            # Show live transcription count
            if st.session_state.rtc_transcriptions:
                st.info(f"📝 Captured {len(st.session_state.rtc_transcriptions)} voice segments so far...")
            
            # Show stream status
            if 'stream_cleared' in st.session_state and st.session_state.stream_cleared:
                st.success("✅ Previous message sent! Stream cleared. Ready for new input.")
                # Reset the flag after showing
                st.session_state.stream_cleared = False
            
        else:
            if st.session_state.rtc_transcriptions:
                st.markdown("""
                <div style="background: #d4edda; color: #155724; padding: 0.5rem; border-radius: 8px; text-align: center; margin: 0.5rem 0;">
                    ✅ Recording stopped. Your voice has been transcribed and is ready to send!
                </div>
                """, unsafe_allow_html=True)
        
        # Display live transcriptions
        if st.session_state.rtc_transcriptions:
            st.markdown("##### 📝 Live Transcription:")
            
            # Show recent transcriptions
            for i, trans in enumerate(st.session_state.rtc_transcriptions[-3:]):  # Show last 3
                st.markdown(f"""
                <div class="rtc-transcription-box">
                    <strong>Segment {len(st.session_state.rtc_transcriptions) - 2 + i}:</strong> {trans['text']}
                </div>
                """, unsafe_allow_html=True)
            
            # Combined transcription for input
            combined_text = " ".join([t['text'] for t in st.session_state.rtc_transcriptions])
            
            st.markdown("##### 💬 Your Complete Message:")
            
            # Show combined text in a text area that user can edit
            final_text = st.text_area(
                "Review and edit your transcribed message:",
                value=combined_text,
                height=120,
                key="rtc_final_input",
                help="You can edit this text before sending if needed"
            )
            
            # Update main user input with the final text (including any edits)
            st.session_state['user_input'] = final_text
            
            return final_text
        else:
            # No transcriptions yet
            st.session_state['user_input'] = ""
            return ""
