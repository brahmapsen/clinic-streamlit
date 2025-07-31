import streamlit as st
import requests
from streamlit_agraph import agraph, Node, Edge, Config

class DashboardHandler:
    def __init__(self, api_url: str = None):
        self.api_url = api_url
    
    def render_personal_info_section(self):
        """Render the personal information input section"""
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
        
        return age, gender, height, weight
    
    def calculate_bmi(self, height: str, weight: str) -> tuple:
        """Calculate BMI and return BMI value and status"""
        try:
            height_val = float(height) if height else 170
            weight_val = float(weight) if weight else 70
            height_m = height_val / 100
            bmi = weight_val / (height_m ** 2)
            bmi_display = f"{bmi:.1f}"
            
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
            
            return bmi_display, bmi_status, bmi_color, bmi
            
        except (ValueError, ZeroDivisionError):
            return "-", "-", "", 0
    
    def render_vitals_section(self, height: str, weight: str):
        """Render the vitals section with BMI calculation"""
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
            bmi_display, bmi_status, bmi_color, bmi_value = self.calculate_bmi(height, weight)
            st.markdown(f"<div style='text-align:center;font-weight:bold;font-size:1.1em'>{bmi_display}</div>", unsafe_allow_html=True)
        
        with status_col:
            st.markdown("Status")
            if bmi_display != "-":
                st.markdown(f"<div style='text-align:center;font-weight:bold;font-size:1.1em'>{bmi_color} {bmi_status}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align:center;'>-</div>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return systolic_bp, diastolic_bp
    
    def render_alerts_section(self):
        """Render the alerts and notifications section"""
        st.markdown("### 🔔 Alerts & Notifications")
        st.markdown('<div class="alert-card">', unsafe_allow_html=True)
        st.info("📱 No new alerts at this time. Check back later for health reminders and updates.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    def save_profile(self, age: str, gender: str, height: str, weight: str, systolic_bp: str, diastolic_bp: str):
        """Save profile data to session state and sync with backend"""
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
            
            if self.api_url:
                profile_payload = {
                    "user_id": st.session_state.user_id,
                    **st.session_state.user_profile
                }
                resp = requests.post(f"{self.api_url}/update_profile", json=profile_payload)
                if resp.status_code == 200:
                    st.success("Profile synced with backend!")
                else:
                    st.error(f"Error syncing profile: {resp.text}")
            else:
                st.success("Profile saved locally!")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    def render_knowledge_graph(self):
        """Render the profile knowledge graph"""
        st.markdown('### 🧠 Profile Knowledge Graph')
        profile = st.session_state.user_profile
        
        # Create nodes and edges for the graph
        nodes = [Node(id="user", label="User", size=30, color="blue")]
        edges = []
        
        for k, v in profile.items():
            nodes.append(Node(id=k, label=f"{k}: {v}", size=20, color="green"))
            edges.append(Edge(source="user", target=k))
        
        # Configure the graph
        config = Config(
            width=400, 
            height=300, 
            directed=False, 
            nodeHighlightBehavior=True, 
            highlightColor="#F7A7A6", 
            collapsible=True
        )
        
        # Render the graph
        agraph(nodes=nodes, edges=edges, config=config)
    
    def render_dashboard_tab(self):
        """Main function to render the entire dashboard tab"""
        # Render personal info section
        age, gender, height, weight = self.render_personal_info_section()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Render vitals section
        systolic_bp, diastolic_bp = self.render_vitals_section(height, weight)
        
        # Render alerts section
        self.render_alerts_section()
        
        # Save profile button
        if st.button("🔄 Save Profile"):
            self.save_profile(age, gender, height, weight, systolic_bp, diastolic_bp)
        
        # Render knowledge graph
        self.render_knowledge_graph()
