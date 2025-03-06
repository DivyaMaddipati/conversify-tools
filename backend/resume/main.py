
import streamlit as st
import google.generativeai as genai
import PyPDF2 as pdf
import json
from dotenv import load_dotenv
import os

# Load environment variables (if available)
load_dotenv()

# Configure Google AI API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCcWfA5IhzK1hZkr5gONwvALAsYN5hOZu8")
genai.configure(api_key=GOOGLE_API_KEY)

# Correct model name
MODEL_NAME = 'gemini-1.5-pro'

# Function to extract text from a PDF file
def extract_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return text.strip()

def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt, 
            generation_config={"temperature": 0.3}  # Controls randomness
        )

        raw_text = response.text.strip()

        # Remove markdown formatting (like ```json ... ```)
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()

        try:
            return json.loads(raw_text)  # Ensure proper JSON format
        except json.JSONDecodeError:
            st.error("⚠️ Gemini AI returned an invalid response. Check raw response above.")
            st.write("🔍 **Raw AI Response (after cleanup):**", raw_text)
            return None
    except Exception as e:
        st.error(f"❌ Gemini API Error: {str(e)}")
        return None

# Streamlit UI Setup
st.title("🔍 Smart ATS - Resume Matching System")
st.text("Upload your resume and enter the job description to get ATS insights.")

# Job Description Input
jd = st.text_area("📄 Paste Job Description", help="Provide the job description for the role.")

# Resume Upload
uploaded_file = st.file_uploader("📂 Upload Resume (PDF)", type=["pdf"], help="Upload your resume in PDF format.")

if uploaded_file and jd:
    resume_text = extract_pdf_text(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Get Matching Percentage"):
            prompt = f"""
            Act as a professional ATS system. Analyze the resume against the job description and provide ONLY a JSON response.

            Resume: {resume_text}
            Job Description: {jd}

            Return ONLY valid JSON in the exact format below:
            ```json
            {{
              "JD Match": "XX%"
            }}
            ```
            Do not include explanations or extra text.
            """
            response = get_gemini_response(prompt)
            if response and "JD Match" in response:
                match_score = response["JD Match"]
                st.subheader(f"✅ Match Score: {match_score}")
            else:
                st.error("⚠️ AI response did not contain a valid 'JD Match' value.")
    
    with col2:
        if st.button("📝 Get Detailed Feedback"):
            prompt = f"""
            Act as a highly skilled ATS system. Analyze the resume against the job description and return a detailed analysis in JSON format.

            Resume: {resume_text}
            Job Description: {jd}

            Return ONLY valid JSON, structured exactly like this:
            ```json
            {{
              "JD Match": "XX%",
              "Missing Keywords": ["keyword1", "keyword2"],
              "Profile Summary": "Summarized analysis of the resume...",
              "Strengths": "Key strengths of the candidate...",
              "Weaknesses": "Areas where the resume could be improved...",
              "Recommend Courses & Resources": "Suggested courses and materials..."
            }}
            ```
            Do not include any explanations, just the raw JSON response.
            """
            response = get_gemini_response(prompt)
            if response and isinstance(response, dict):
                st.subheader("📊 Resume Analysis Report")
                st.write("🔹 **Match Score**: ", response.get("JD Match", "N/A"))
                st.write("🔹 **Missing Keywords**: ", ", ".join(response.get("Missing Keywords", [])))
                st.write("🔹 **Profile Summary**: ", response.get("Profile Summary", "N/A"))
                st.write("🔹 **Strengths**: ", response.get("Strengths", "N/A"))
                st.write("🔹 **Weaknesses**: ", response.get("Weaknesses", "N/A"))
                st.write("🔹 **Recommended Courses & Resources**: ", response.get("Recommend Courses & Resources", "N/A"))
            else:
                st.error("⚠️ AI response did not return a valid JSON object.")

st.text("Powered by Google Generative AI 🤖")
