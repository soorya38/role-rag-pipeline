import streamlit as st
import requests
import json
import time

# --- Setup App Config ---
st.set_page_config(
    page_title="Role Matcher (RAG)",
    page_icon="🔍",
    layout="wide",
)

st.title("🧑‍💻 Role Matcher")
st.subheader("Match Candidate Roles to Job Descriptions via RAG & LLM Analysis")

API_URL = "http://127.0.0.1:8000/api/match"

with st.sidebar:
    st.header("⚙️ Configuration")
    top_k = st.slider("Top K Candidates to Retrieve", min_value=1, max_value=20, value=5)
    
    st.markdown("---")
    st.markdown("""
    **How to use:**
    1. Paste the job description in the main area.
    2. Click 'Analyze and Match Roles'.
    3. The application queries the vector database for candidates and runs an LLM evaluation on their fit.
    """)

# Main Content Area
job_description = st.text_area(
    "Paste the Job Description Here:",
    height=300,
    placeholder="e.g. We are looking for a Senior Python Developer with experience in FastAPI and Docker..."
)

if st.button("Analyze and Match Roles", type="primary"):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        with st.spinner("🔍 Retrieving candidate roles and running LLM analysis. This may take a minute..."):
            start_time = time.time()
            try:
                # Prepare payload
                payload = {
                    "query": job_description.strip(),
                    "top_k": top_k
                }
                
                # Make POST request to the backend
                response = requests.post(API_URL, json=payload)
                response.raise_for_status() 
                
                data = response.json()
                results = data.get("results", [])
                analysis = data.get("analysis", {})
                
                elapsed_time = round(time.time() - start_time, 2)
                st.success(f"Analysis completed in {elapsed_time}s")
                
                # --- Rendering the Results ---
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.header("📊 Executive Summary & Ranking")
                    st.info(analysis.get("summary_and_ranking", "No summary available."))
                    
                    st.header("📝 Cover Letter Draft")
                    st.success(analysis.get("cover_letter", "No cover letter available."))
                    
                with col2:
                    st.header("🔍 Detailed Fit Analysis")
                    st.warning(analysis.get("fit_analysis", "No fit analysis available."))
                
                st.markdown("---")
                st.header("🗂️ Raw Retrieved Candidate Roles")
                
                if not results:
                    st.info("No candidates retrived.")
                else:
                    for i, candidate in enumerate(results, 1):
                        with st.expander(f"[{i}] {candidate.get('role')} (Score: {candidate.get('score')})"):
                            st.markdown(f"**Tools:** {candidate.get('tools', 'N/A')}")
                            st.markdown(f"**Projects:** {candidate.get('projects', 'N/A')}")
                            
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the backend. Is the FastAPI server running on `http://127.0.0.1:8000`?")
            except requests.exceptions.HTTPError as e:
                st.error(f"Backend returned an error: {e}")
                st.text(response.text)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
