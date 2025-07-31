import streamlit as st
import requests
import json
from typing import List, Dict, Optional
import random

class ProviderFinder:
    def __init__(self):
        self.insurance_providers = [
            "Blue Cross Blue Shield",
            "Aetna",
            "Cigna",
            "UnitedHealthcare",
            "Humana",
            "Kaiser Permanente",
            "Anthem",
            "Molina Healthcare"
        ]
        
        self.plan_types = ["PPO", "HMO", "EPO", "POS"]
        self.plan_levels = ["Bronze", "Silver", "Gold", "Platinum"]
        
        self.provider_specialties = [
            "Primary Care Physician",
            "Cardiologist (Heart Specialist)",
            "Orthopedic Surgeon",
            "Nutritionist/Dietitian",
            "Dermatologist",
            "Neurologist",
            "Psychiatrist",
            "Gynecologist",
            "Pediatrician",
            "Endocrinologist",
            "Gastroenterologist",
            "Ophthalmologist",
            "Urologist",
            "Oncologist",
            "Pulmonologist"
        ]
        
        # Mock provider data for demonstration
        self.mock_providers = self._generate_mock_providers()
    
    def _generate_mock_providers(self) -> List[Dict]:
        """Generate mock provider data for demonstration"""
        providers = []
        
        # Sample provider names and locations
        sample_data = [
            {"name": "Dr. Sarah Johnson", "specialty": "Primary Care Physician", "location": "Downtown Medical Center"},
            {"name": "Dr. Michael Chen", "specialty": "Cardiologist (Heart Specialist)", "location": "Heart & Vascular Institute"},
            {"name": "Dr. Emily Rodriguez", "specialty": "Orthopedic Surgeon", "location": "Sports Medicine Clinic"},
            {"name": "Lisa Thompson, RD", "specialty": "Nutritionist/Dietitian", "location": "Wellness Center"},
            {"name": "Dr. David Kim", "specialty": "Dermatologist", "location": "Skin Care Specialists"},
            {"name": "Dr. Jennifer Walsh", "specialty": "Neurologist", "location": "Brain & Spine Center"},
            {"name": "Dr. Robert Martinez", "specialty": "Psychiatrist", "location": "Mental Health Associates"},
            {"name": "Dr. Amanda Foster", "specialty": "Gynecologist", "location": "Women's Health Center"},
            {"name": "Dr. James Wilson", "specialty": "Pediatrician", "location": "Children's Medical Group"},
            {"name": "Dr. Maria Gonzalez", "specialty": "Endocrinologist", "location": "Diabetes & Hormone Center"},
        ]
        
        for i, provider_data in enumerate(sample_data):
            # Generate multiple providers per specialty
            for j in range(2, 5):  # 2-4 providers per specialty
                provider = {
                    "id": f"prov_{i}_{j}",
                    "name": provider_data["name"] if j == 2 else f"Dr. {['Alex', 'Jordan', 'Taylor', 'Casey'][j-2]} {['Smith', 'Brown', 'Davis', 'Miller'][j-2]}",
                    "specialty": provider_data["specialty"],
                    "location": provider_data["location"],
                    "address": f"{random.randint(100, 9999)} {random.choice(['Main St', 'Oak Ave', 'Pine Rd', 'Elm Dr'])}, City, State {random.randint(10000, 99999)}",
                    "phone": f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}",
                    "rating": round(random.uniform(3.5, 5.0), 1),
                    "accepting_patients": random.choice([True, True, True, False]),  # 75% accepting
                    "distance": round(random.uniform(0.5, 15.0), 1),
                    "insurance_networks": random.sample(self.insurance_providers, random.randint(3, 6)),
                    "plan_types_accepted": random.sample(self.plan_types, random.randint(2, 4)),
                    "languages": random.sample(["English", "Spanish", "French", "Mandarin", "Korean"], random.randint(1, 3))
                }
                providers.append(provider)
        
        return providers
    
    def search_providers(self, specialty: str, insurance_provider: str, plan_type: str, plan_level: str, location: str = "") -> List[Dict]:
        """Search for providers based on criteria"""
        
        # Filter providers based on criteria
        filtered_providers = []
        
        for provider in self.mock_providers:
            # Check specialty match
            if specialty.lower() not in provider["specialty"].lower():
                continue
            
            # Check insurance network
            if insurance_provider not in provider["insurance_networks"]:
                continue
            
            # Check plan type
            if plan_type not in provider["plan_types_accepted"]:
                continue
            
            # Add provider to results
            filtered_providers.append(provider)
        
        # Sort by rating and distance
        filtered_providers.sort(key=lambda x: (-x["rating"], x["distance"]))
        
        return filtered_providers[:10]  # Return top 10 results
    
    def get_provider_details(self, provider_id: str) -> Optional[Dict]:
        """Get detailed information about a specific provider"""
        for provider in self.mock_providers:
            if provider["id"] == provider_id:
                return provider
        return None
    
    def render_insurance_form(self) -> Dict:
        """Render insurance information form"""
        st.markdown("#### 🏥 Insurance Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            insurance_provider = st.selectbox(
                "Insurance Provider",
                options=self.insurance_providers,
                key="insurance_provider"
            )
            
            plan_type = st.selectbox(
                "Plan Type",
                options=self.plan_types,
                key="plan_type"
            )
        
        with col2:
            plan_level = st.selectbox(
                "Plan Level",
                options=self.plan_levels,
                key="plan_level"
            )
            
            location = st.text_input(
                "Location (City, State)",
                placeholder="e.g., San Francisco, CA",
                key="provider_location"
            )
        
        return {
            "insurance_provider": insurance_provider,
            "plan_type": plan_type,
            "plan_level": plan_level,
            "location": location
        }
    
    def render_specialty_selection(self) -> str:
        """Render provider specialty selection"""
        st.markdown("#### 👨‍⚕️ Provider Type")
        
        specialty = st.selectbox(
            "What type of provider are you looking for?",
            options=self.provider_specialties,
            key="provider_specialty"
        )
        
        return specialty
    
    def render_provider_results(self, providers: List[Dict]):
        """Render search results"""
        if not providers:
            st.warning("No providers found matching your criteria. Try adjusting your search parameters.")
            return
        
        st.markdown(f"#### 📋 Found {len(providers)} Provider(s)")
        
        for provider in providers:
            with st.container():
                st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div>
                            <h4 style="margin: 0 0 0.5rem 0; color: #333;">{provider['name']}</h4>
                            <p style="margin: 0 0 0.25rem 0; color: #666; font-weight: 600;">{provider['specialty']}</p>
                            <p style="margin: 0 0 0.25rem 0; color: #888;">{provider['location']}</p>
                            <p style="margin: 0 0 0.25rem 0; color: #888; font-size: 0.9rem;">{provider['address']}</p>
                            <p style="margin: 0 0 0.5rem 0; color: #888; font-size: 0.9rem;">📞 {provider['phone']}</p>
                        </div>
                        <div style="text-align: right;">
                            <div style="background: {'#4CAF50' if provider['accepting_patients'] else '#f44336'}; color: white; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; margin-bottom: 0.5rem;">
                                {'✅ Accepting Patients' if provider['accepting_patients'] else '❌ Not Accepting'}
                            </div>
                            <div style="color: #666; font-size: 0.9rem;">⭐ {provider['rating']}/5.0</div>
                            <div style="color: #666; font-size: 0.9rem;">📍 {provider['distance']} miles</div>
                        </div>
                    </div>
                    <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid #eee;">
                        <p style="margin: 0; color: #666; font-size: 0.9rem;">
                            <strong>Languages:</strong> {', '.join(provider['languages'])}
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"📞 Call", key=f"call_{provider['id']}"):
                        st.info(f"Calling {provider['phone']}...")
                with col2:
                    if st.button(f"📅 Book Appointment", key=f"book_{provider['id']}"):
                        st.success(f"Appointment booking for {provider['name']} - Feature coming soon!")
                with col3:
                    if st.button(f"ℹ️ More Info", key=f"info_{provider['id']}"):
                        self.show_provider_details(provider)
    
    def show_provider_details(self, provider: Dict):
        """Show detailed provider information"""
        with st.expander(f"Details for {provider['name']}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Specialty:** {provider['specialty']}  
                **Location:** {provider['location']}  
                **Address:** {provider['address']}  
                **Phone:** {provider['phone']}  
                **Rating:** ⭐ {provider['rating']}/5.0  
                **Distance:** 📍 {provider['distance']} miles
                """)
            
            with col2:
                st.markdown(f"""
                **Accepting Patients:** {'✅ Yes' if provider['accepting_patients'] else '❌ No'}  
                **Languages:** {', '.join(provider['languages'])}  
                **Insurance Networks:** {', '.join(provider['insurance_networks'][:3])}{'...' if len(provider['insurance_networks']) > 3 else ''}  
                **Plan Types:** {', '.join(provider['plan_types_accepted'])}
                """)
    
    def render_provider_tab(self):
        """Main function to render the entire provider tab"""
        st.markdown("### 🔍 Find Healthcare Providers")
        st.markdown("Find providers in your insurance network based on your plan and specialty needs.")
        
        # Insurance form
        insurance_info = self.render_insurance_form()
        
        # Specialty selection
        specialty = self.render_specialty_selection()
        
        # Search button
        if st.button("🔍 Search Providers", type="primary"):
            with st.spinner("Searching for providers..."):
                providers = self.search_providers(
                    specialty=specialty,
                    insurance_provider=insurance_info["insurance_provider"],
                    plan_type=insurance_info["plan_type"],
                    plan_level=insurance_info["plan_level"],
                    location=insurance_info["location"]
                )
                
                # Store results in session state
                st.session_state['provider_search_results'] = providers
                st.session_state['last_search_params'] = {
                    'specialty': specialty,
                    'insurance': insurance_info["insurance_provider"],
                    'plan_type': insurance_info["plan_type"],
                    'plan_level': insurance_info["plan_level"]
                }
        
        # Display results if available
        if 'provider_search_results' in st.session_state:
            st.markdown("---")
            self.render_provider_results(st.session_state['provider_search_results'])
        
        # Additional information
        st.markdown("---")
        st.markdown("#### 💡 Tips for Finding Providers")
        st.info("""
        - **Verify Coverage**: Always confirm with your insurance before scheduling
        - **Check Availability**: Call ahead to confirm the provider is accepting new patients
        - **Consider Distance**: Factor in travel time for regular appointments
        - **Read Reviews**: Look up provider reviews on healthcare websites
        - **Specialist Referrals**: Some specialists may require a referral from your primary care doctor
        """)
