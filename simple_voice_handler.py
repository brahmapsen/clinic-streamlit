import streamlit as st
import speech_recognition as sr
import io
import tempfile
import os
import requests

class SimpleVoiceRecorder:
    def __init__(self, groq_api_key: str = None):
        self.groq_api_key = groq_api_key
        self.recognizer = sr.Recognizer()
        
    def transcribe_with_groq(self, audio_bytes):
        """Transcribe audio using Groq API"""
        if not self.groq_api_key:
            return None, "Groq API key not provided"
            
        try:
            files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
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
                timeout=10
            )
            
            if response.status_code == 200:
                return response.text.strip(), None
            else:
                return None, f"Groq API error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return None, f"Error transcribing with Groq: {str(e)}"
    
    def transcribe_with_speech_recognition(self, audio_bytes):
        """Transcribe audio using local speech_recognition library"""
        try:
            # Save audio bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            try:
                # Load audio file
                with sr.AudioFile(tmp_file_path) as source:
                    audio = self.recognizer.record(source)
                
                # Transcribe using Google Speech Recognition (free)
                text = self.recognizer.recognize_google(audio)
                return text, None
                
            except sr.UnknownValueError:
                return None, "Could not understand audio"
            except sr.RequestError as e:
                return None, f"Speech recognition error: {e}"
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
        except Exception as e:
            return None, f"Error processing audio: {str(e)}"
    
    def on_audio_change(self):
        """Callback when audio recording changes"""
        audio_file = st.session_state.get('voice_audio_input')
        if audio_file:
            # Get transcription method from session state
            # transcription_method = st.session_state.get('transcription_method', 'Groq API (Better Quality)')
            transcription_method = "Local Speech Recognition (Free)"
            with st.spinner("Transcribing audio..."):
                audio_bytes = audio_file.read()
                
                if transcription_method == "Local Speech Recognition (Free)":
                    text, error = self.transcribe_with_speech_recognition(audio_bytes)
                else:
                    text, error = self.transcribe_with_groq(audio_bytes)
                
                if text:
                    st.session_state['user_input'] = text
                    st.session_state['transcription_result'] = text
                    st.session_state['transcription_error'] = None
                else:
                    st.session_state['transcription_error'] = error
                    st.session_state['transcription_result'] = None

    def render_voice_interface(self):
        """Render the simple voice recording interface using st.audio_input"""
        
        # Transcription method selection--BPS
        # transcription_method = st.radio(
        #     "Choose transcription method:",
        #     ["Local Speech Recognition (Free)", "Groq API (Better Quality)"],
        #     horizontal=True,
        #     key="transcription_method"
        # )
        
        # Audio input widget with callback
        audio_file = st.audio_input(
            "Record your message", 
            key="voice_audio_input",
            on_change=self.on_audio_change
        )
        
        # Show transcription results
        if 'transcription_result' in st.session_state and st.session_state['transcription_result']:
            st.success("✅ Transcription completed!")
            st.markdown("**Transcribed Text:**")
            transcribed_text = st.session_state['transcription_result']
            st.text_area("Transcribed Text", value=transcribed_text, height=100, key="transcription_display", disabled=True, label_visibility="collapsed")
            
            # Clear button
            if st.button("🗑️ Clear Transcription"):
                st.session_state['user_input'] = ""
                st.session_state['transcription_result'] = None
                st.session_state['transcription_error'] = None
            
            return transcribed_text
        
        elif 'transcription_error' in st.session_state and st.session_state['transcription_error']:
            st.error(f"❌ Transcription failed: {st.session_state['transcription_error']}")
        
        # Show current transcribed text if available
        current_text = st.session_state.get('user_input', '')
        if current_text and not audio_file:
            st.markdown("**Current transcribed text:**")
            st.text_area("Current Transcription", value=current_text, height=100, key="current_transcription", disabled=True, label_visibility="collapsed")
        
        return current_text
