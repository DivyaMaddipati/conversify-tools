import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure Google Generative AI
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    st.error("Failed to configure Generative AI. Please check your API key.")

# Function to interact with the Gemini API
def get_gemini_response(input_text):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input_text)
    return response.text

# Function to extract text from a PDF file
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

# Streamlit app
st.title("Smart ATS")
st.text("Enhance Your Resume with ATS Insights")

# Job Description Input
jd = st.text_area("Paste the Job Description", help="Provide the job description for the role.")

# Resume Upload
uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Upload your resume in PDF format.")

# Buttons for functionality
if st.button("Get Matching Percentage"):
    if uploaded_file is not None and jd:
        resume_text = input_pdf_text(uploaded_file)
        matching_prompt = f"""
        Hey Act Like a skilled or very experienced ATS. Your task is to evaluate the resume based on the given job description. 
        Only provide the JD Match percentage as a response.

        Resume: {resume_text}
        Job Description: {jd}

        Provide response in the format:
        {{"JD Match": "%"}}
        """
        try:
            response = get_gemini_response(matching_prompt)
            st.subheader("Matching Percentage")
            st.write(response)
        except Exception as e:
            st.error("Failed to process the request. Please try again later.")
    else:
        st.warning("Please upload a resume and provide a job description.")

if st.button("Get Detailed Feedback"):
    if uploaded_file is not None and jd:
        resume_text = input_pdf_text(uploaded_file)
        feedback_prompt = f"""
        Hey Act Like a skilled or very experienced ATS. Your task is to evaluate the resume based on the given job description. 
        Provide detailed feedback including:
        1. JD Match (%): Assign the percentage matching based on the Job Description (JD) and the resume provided.
        2. Missing Keywords: Identify missing keywords with high accuracy and relevance to the JD.
        3. Profile Summary: Summarize the profile's strengths and alignment with the JD.
        4. Strengths: Highlight the key strengths of the candidate based on the resume.
        5. Weaknesses: Point out weaknesses or areas that need improvement based on the JD.
        6. Recommend Courses & Resources: Suggest relevant courses or resources to improve the profile and match the JD better.

        Resume: {resume_text}
        Job Description: {jd}

        Provide the response in this format:
        {{
          "JD Match": "%",
          "Missing Keywords": [],
          "Profile Summary": "",
          "Strengths": "",
          "Weaknesses": "",
          "Recommend Courses & Resources": ""
        }}
        """
        try:
            response = get_gemini_response(feedback_prompt)
            st.subheader("Detailed Feedback")
            st.write(response)
        except Exception as e:
            st.error("Failed to process the request. Please try again later.")
    else:
        st.warning("Please upload a resume and provide a job description.")

st.text("\nPowered by Google Generative AI")
