import streamlit as st
import requests
import json
from datetime import datetime
import uuid
import os
import re
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AGENT_API_URL = os.getenv("AGENT_API_URL")

# Add import for streamlit-agraph at the top
from streamlit_agraph import agraph, Node, Edge, Config
# from st_audiorec import st_audiorec
from simple_voice_handler import SimpleVoiceRecorder

# Configure Streamlit page for mobile-like experience
st.set_page_config(
    page_title="CIP Mobile App",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile-like styling
st.markdown("""
<style>
    /* Mobile app container */
    .main .block-container {
        max-width: 400px !important;
        padding: 1rem !important;
        margin: 0 auto !important;
    }
    
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .tab-content {
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .profile-card {
        background: white;
        padding: 0.75rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 0.75rem;
    }
    
    .profile-card h4 {
        margin-bottom: 0.5rem !important;
        font-size: 1.1rem !important;
    }
    
    .alert-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 0.75rem;
        border-radius: 10px;
        margin-bottom: 0.75rem;
    }
    
    .chat-message {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 15px;
        max-width: 85%;
    }
    
    .user-message {
        background-color: #007bff;
        color: white;
        margin-left: auto;
        text-align: right;
    }
    
    .assistant-message {
        background-color: #e9ecef;
        color: #333;
        margin-right: auto;
    }
    
    .clinic-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 0.75rem;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 0.75rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
    }
    
    /* Compact input styling */
    .stTextInput > div > div > input {
        padding: 0.5rem !important;
        font-size: 14px !important;
    }
    
    .stSelectbox > div > div > div {
        padding: 0.5rem !important;
        font-size: 14px !important;
    }
    
    .stTextArea > div > div > textarea {
        padding: 0.5rem !important;
        font-size: 14px !important;
    }
    
    /* Compact input fields */
    .compact-input input, .compact-input select {
        width: 70px !important;
        min-width: 50px !important;
        max-width: 90px !important;
        padding: 0.25rem 0.5rem !important;
        font-size: 13px !important;
        margin-right: 4px !important;
    }
    .compact-row {
        display: flex;
        flex-direction: row;
        gap: 8px;
        margin-bottom: 0.2rem !important;
    }
    .dashboard-title {
        margin-bottom: 0.3rem !important;
        margin-top: 0.2rem !important;
    }
    .compact-section-header {
        margin-top: 0.1rem !important;
    }
    
    /* Reduce spacing between elements */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* Custom tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 6px;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 0px 16px;
        background-color: white;
        border-radius: 8px;
        color: #666;
        font-weight: 600;
        font-size: 14px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: 2px solid #667eea;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(0,0,0,0.15);
    }
    
    .stTabs [aria-selected="true"]:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-1px);
    }
    
    /* Section headers */
    h3 {
        font-size: 1.2rem !important;
        margin-bottom: 0.5rem !important;
        margin-top: 1rem !important;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 8px;
        padding: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    h3, .compact-section-header {
        font-size: 1.08rem !important;
        margin-bottom: 0.2rem !important;
        margin-top: 0.2rem !important;
        font-weight: 600 !important;
        line-height: 1.2 !important;
    }
    
    /* Real-time voice interface styling */
    .rtc-recording-active {
        background: linear-gradient(90deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }
    
    .rtc-recording-inactive {
        background: #f8f9fa;
        color: #666;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
        border: 2px dashed #ddd;
    }
    
    .rtc-transcription-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* WebRTC component styling */
    .stWebRtc {
        border-radius: 8px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'age': 25,
        'gender': 'Male',
        'height': 170,
        'weight': 70,
        'systolic_bp': 120,
        'diastolic_bp': 80
    }

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'api_url' not in st.session_state:
    st.session_state.api_url = AGENT_API_URL 

# Add persistent user_id for conversation continuity
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{str(uuid.uuid4())[:8]}"

# Add new tab for advanced tools
if 'advanced_tab' not in st.session_state:
    st.session_state.advanced_tab = False

# Define the compact header as a variable for reuse
compact_header = """
<div style='text-align: center; color: #222; padding: 0.5rem 0 0.5rem 0; margin-bottom: 0.5rem; font-size: 1.08rem; font-weight: bold;'>
    🏥 <b>Clinic-In-Pocket</b> &mdash; <span style='font-weight:bold;'>Your Primary Care Companion</span><br/>
    <span style='font-size:0.92em; font-weight:normal; color:#666;'>For testing purposes only | Always consult healthcare professionals for medical advice</span>
</div>
"""

# Place the header at the very top, before the tabs
st.markdown(compact_header, unsafe_allow_html=True)

# Now define the tabs (remove header from inside each tab)
tab1, tab2, tab3 = st.tabs(["📊 **Dashboard**", "💬 **Ask**", "🏥 **Find a Clinic**"])

# Dashboard Tab
with tab1:
    # st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("#### Personal Info")

    # Four fields in one row, each 1/4 width
    age_col, gender_col, height_col, weight_col = st.columns(4)
    with age_col:
        age = st.text_input(
            "Age", 
            value=str(st.session_state.user_profile['age']),
            key="profile_age",
            placeholder="25"
        )
    with gender_col:
        gender = st.selectbox(
            "Gender", 
            options=["Male", "Female", "Other"],
            index=["Male", "Female", "Other"].index(st.session_state.user_profile['gender']),
            key="profile_gender"
        )
    with height_col:
        height = st.text_input(
            "Height (cm)", 
            value=str(st.session_state.user_profile['height']),
            key="profile_height",
            placeholder="170"
        )
    with weight_col:
        weight = st.text_input(
            "Weight (kg)", 
            value=str(st.session_state.user_profile['weight']),
            key="profile_weight",
            placeholder="70"
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Vitals Section
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.markdown("#### Vitals")
    # Four fields in one row: Systolic, Diastolic, BMI, Status
    sys_col, dias_col, bmi_col, status_col = st.columns(4)
    with sys_col:
        st.markdown("Systolic")
        systolic_bp = st.text_input(
            "Systolic", 
            value=str(st.session_state.user_profile['systolic_bp']),
            key="profile_systolic",
            placeholder="120",
            label_visibility="collapsed"
        )
    with dias_col:
        st.markdown("Diastolic")
        diastolic_bp = st.text_input(
            "Diastolic", 
            value=str(st.session_state.user_profile['diastolic_bp']),
            key="profile_diastolic",
            placeholder="80",
            label_visibility="collapsed"
        )
    with bmi_col:
        st.markdown("BMI")
        try:
            height_val = float(height) if height else 170
            weight_val = float(weight) if weight else 70
            height_m = height_val / 100
            bmi = weight_val / (height_m ** 2)
            bmi_display = f"{bmi:.1f}"
        except (ValueError, ZeroDivisionError):
            bmi_display = "-"
        st.markdown(f"<div style='text-align:center;font-weight:bold;font-size:1.1em'>{bmi_display}</div>", unsafe_allow_html=True)
    with status_col:
        st.markdown("Status")
        try:
            if bmi < 18.5:
                bmi_status = "Underweight"
                bmi_color = "🔵"
            elif bmi < 25:
                bmi_status = "Normal"
                bmi_color = "🟢"
            elif bmi < 30:
                bmi_status = "Overweight"
                bmi_color = "🟡"
            else:
                bmi_status = "Obese"
                bmi_color = "🔴"
            st.markdown(f"<div style='text-align:center;font-weight:bold;font-size:1.1em'>{bmi_color} {bmi_status}</div>", unsafe_allow_html=True)
        except:
            st.markdown("<div style='text-align:center;'>-</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Alerts & Notifications Section
    st.markdown("### 🔔 Alerts & Notifications")
    st.markdown('<div class="alert-card">', unsafe_allow_html=True)
    st.info("📱 No new alerts at this time. Check back later for health reminders and updates.")
    st.markdown('</div>', unsafe_allow_html=True)
    

    # Add profile sync button in Dashboard tab
    if st.button("🔄 Save Profile"):
        try:
            # Update session state with current form values
            st.session_state.user_profile.update({
                'age': int(age) if age.isdigit() else st.session_state.user_profile['age'],
                'gender': gender,
                'height': int(height) if height.isdigit() else st.session_state.user_profile['height'],
                'weight': int(weight) if weight.isdigit() else st.session_state.user_profile['weight'],
                'systolic_bp': int(systolic_bp) if systolic_bp.isdigit() else st.session_state.user_profile['systolic_bp'],
                'diastolic_bp': int(diastolic_bp) if diastolic_bp.isdigit() else st.session_state.user_profile['diastolic_bp']
            })
            
            profile_payload = {
                "user_id": st.session_state.user_id,
                **st.session_state.user_profile
            }
            resp = requests.post(f"{st.session_state.api_url}/update_profile", json=profile_payload)
            if resp.status_code == 200:
                st.success("Profile synced with backend!")
            else:
                st.error(f"Error syncing profile: {resp.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Knowledge Graph Section
    st.markdown('### 🧠 Profile Knowledge Graph')
    profile = st.session_state.user_profile
    nodes = [Node(id="user", label="User", size=30, color="blue")]
    edges = []
    for k, v in profile.items():
        nodes.append(Node(id=k, label=f"{k}: {v}", size=20, color="green"))
        edges.append(Edge(source="user", target=k))
    config = Config(width=400, height=300, directed=False, nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True)
    agraph(nodes=nodes, edges=edges, config=config)


def get_tts_audio(text, model_name="playai-tts", voice="Fritz-PlayAI", response_format="wav"):
    """
    Calls the Groq TTS API to synthesize speech from text.
    Returns (audio_bytes, error_message). If successful, error_message is None.
    """
    # groq_api_key = os.getenv("GROQ_API_KEY")
    data = {
        "model": model_name,
        "input": text,
        "voice": voice,
        "response_format": response_format
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    response = requests.post(
        "https://api.groq.com/openai/v1/audio/speech",
        json=data,
        headers=headers
    )
    if response.status_code == 200:
        return response.content, None
    else:
        return None, f"TTS failed: {response.text}"

def transcribe_audio_with_groq(audio_to_use, audio_label, audio_file=None):
    """
    Calls the Groq Speech-to-Text API and returns the transcribed text or error.
    audio_to_use: tuple for recorded audio or file-like for uploaded audio
    audio_label: 'Recorded audio' or 'Uploaded audio'
    audio_file: the uploaded file (needed for Uploaded audio)
    """
    if audio_label == "Recorded audio":
        files = {"file": audio_to_use}
    else:
        files = {"file": audio_file}
    data = {
        "model": "whisper-large-v3-turbo",
        "response_format": "text"
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    response = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        files=files,
        data=data,
        headers=headers
    )
    if response.status_code == 200:
        return response.text, None
    else:
        return None, f"Transcription failed: {response.text}"

# Ask Tab
with tab2:
    # st.markdown(compact_header, unsafe_allow_html=True)
    # st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 💬 Ask Your Health Question")

    # Quick question buttons in one row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🤒 Fever & Symptoms"):
            user_message = "I have a fever and feeling unwell. What should I do?"
    with col2:
        if st.button("💊 Medication Help"):
            user_message = "I need help understanding my medication dosage."
    with col3:
        if st.button("🏃‍♂️ Exercise Advice"):
            user_message = "What exercises are safe for my current health condition?"
    with col4:
        if st.button("🍎 Nutrition Tips"):
            user_message = "Can you give me nutrition advice based on my profile?"

    # --- Toggle between Text and Voice Input ---
    mode = st.radio("Choose input mode:", ["Text", "Voice Recording"], horizontal=True)
    user_message = ""
    
    if mode == "Text":
        user_message = st.text_area(
            "Type your health question here:",
            placeholder="e.g., I have been feeling tired and have a headache for the past 2 days. What should I do?",
            height=100,
            key="user_input"
        )
    else:  # Voice Recording mode
        # Initialize simple voice recorder
        if 'voice_recorder' not in st.session_state:
            st.session_state.voice_recorder = SimpleVoiceRecorder(GROQ_API_KEY)
        
        # Render the simple voice interface
        transcribed_text = st.session_state.voice_recorder.render_voice_interface()
        
        # Get the transcribed text from session state (set by voice recorder)
        user_message = st.session_state.get('user_input', '')
        
        # Show current input status
        if user_message and user_message.strip():
            st.success(f"✅ Voice input ready: {len(user_message.split())} words captured")

    # Chat Interface
    # st.markdown("#### 💭 Chat with Health Assistant")
    
    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)

    # Show Text-To-Speech(TTS) for last assistant response in Voice modes
    if mode in ["Voice Recording"] and st.session_state.chat_history:
        # Find the last assistant message
        last_assistant = next((m for m in reversed(st.session_state.chat_history) if m['role'] == 'assistant'), None)
        if last_assistant and isinstance(last_assistant['content'], str):
            # Extract first two sentences
            sentences = re.split(r'(?<=[.!?]) +', last_assistant['content'])
            tts_text = ' '.join(sentences[:2]).strip()
            if tts_text:
                if st.button("🔊 Play Response Audio"):
                    with st.spinner("Generating audio..."):
                        audio_bytes, tts_error = get_tts_audio(tts_text, model_name="playai-tts")
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/wav")
                            st.success("Audio ready!")
                        else:
                            st.error(tts_error)
    
    
    # Send message button - always enabled
    if st.button("📤 Send Message", type="primary"):
        if user_message.strip():
            # Add user message to chat
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_message
            })
            
            # Prepare API request - backend handles conversation history via user_id
            payload = {
                "user_id": st.session_state.user_id,
                "message": user_message,
                "language": "en"
            }
            
            try:
                with st.spinner("🤔 Thinking..."):
                    response = requests.post(
                        f"{st.session_state.api_url}/triage",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=30
                    )
                
                if response.status_code == 200:
                    # Try to parse for severity, recommendation, suggested action, clarifying questions
                    try:
                        # Try to parse as JSON if backend returns structured response
                        response_data = response.json()
                        assistant_response = response_data.get("response", "No response received")
                        # Try to extract structured info if present
                        if isinstance(assistant_response, dict):
                            st.markdown(f"**Severity:** {assistant_response.get('severity', 'N/A')}")
                            st.markdown(f"**Recommendation:** {assistant_response.get('recommendation', '')}")
                            st.markdown(f"**Suggested Action:** {assistant_response.get('suggested_action', '')}")
                            if assistant_response.get('clarifying_questions'):
                                st.markdown("**Clarifying Questions:**")
                                for q in assistant_response['clarifying_questions']:
                                    st.markdown(f"- {q}")
                        else:
                            st.markdown(assistant_response)
                    except Exception:
                        st.markdown(assistant_response)
                    
                    # Add assistant response to chat
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': assistant_response
                    })
                    
                    st.success("✅ Response received!")
                    st.rerun()
                    
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    if response.text:
                        st.error(response.text)
                        
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the API. Make sure the FastAPI server is running.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Clear chat button
    if st.session_state.chat_history and st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    # --- Advanced Tools (moved here) ---
    st.markdown("---")
    st.markdown("#### 🛠️ Advanced Agent Tools")

    st.markdown("##### 🔬 Symptom Extraction")
    symptom_text = st.text_area("Enter text to extract symptoms:", key="symptom_extract_input")
    if st.button("Extract Symptoms"):
        if symptom_text.strip():
            try:
                resp = requests.post(f"{st.session_state.api_url}/extract_symptoms", json={"user_id": st.session_state.user_id, "message": symptom_text})
                if resp.status_code == 200:
                    symptoms = resp.json().get("symptoms", [])
                    st.success(f"Extracted Symptoms: {', '.join(symptoms)}")
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.markdown("##### ❓ Clarifying Questions")
    clar_text = st.text_area("Enter symptoms (comma separated) or text:", key="clarify_input")
    if st.button("Get Clarifying Questions"):
        if clar_text.strip():
            try:
                # Try to parse as list, else treat as text
                if ',' in clar_text:
                    symptoms = [s.strip() for s in clar_text.split(',') if s.strip()]
                    payload = {"user_id": st.session_state.user_id, "message": clar_text, "symptoms": symptoms}
                else:
                    payload = {"user_id": st.session_state.user_id, "message": clar_text}
                resp = requests.post(f"{st.session_state.api_url}/clarifying_questions", json=payload)
                if resp.status_code == 200:
                    questions = resp.json().get("questions", [])
                    st.success("\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)]))
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

    # API Configuration (collapsible)
    with st.expander("⚙️ API Settings"):
        api_url = st.text_input(
            "API Base URL", 
            value=st.session_state.api_url,
            help="Base URL of your CIP FastAPI application"
        )
        st.session_state.api_url = api_url
        
        # Test connection
        if st.button("🔍 Test Connection"):
            try:
                response = requests.get(f"{api_url}/")
                if response.status_code == 200:
                    st.success("✅ Connection successful!")
                else:
                    st.error(f"❌ Connection failed: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")
    

# Find a Clinic Tab
with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 🏥 Find a Clinic")
    
    st.markdown('<div class="clinic-card">', unsafe_allow_html=True)
    st.markdown("""
    #### 🚨 Need Immediate Medical Attention?
    
    **For emergencies, please contact:**
    - 🚑 **Emergency Services: 911**
    - 🏥 **Local Emergency Room**
    
    #### 🔍 Find Healthcare Near You
    
    We're working on integrating clinic finder functionality. 
    For now, please:
    
    1. **Contact your primary care physician**
    2. **Visit your local urgent care center**
    3. **Use online healthcare directories**
    4. **Call your insurance provider for in-network options**
    
    #### 📞 Telehealth Options
    
    Consider these telehealth services:
    - Video consultations with licensed doctors
    - Online prescription services
    - Mental health support
    - Specialist referrals
    
    #### 🆘 When to Seek Immediate Care
    
    - Chest pain or difficulty breathing
    - Severe allergic reactions
    - High fever (over 103°F)
    - Severe injuries or bleeding
    - Signs of stroke or heart attack
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Placeholder for future functionality
    st.info("🚧 **Coming Soon:** Interactive clinic finder with maps, reviews, and appointment booking!")
    
    st.markdown('</div>', unsafe_allow_html=True)
