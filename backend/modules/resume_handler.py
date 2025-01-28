from flask import jsonify
import PyPDF2 as pdf
import google.generativeai as genai
import json

def extract_pdf_text(file):
    """Extracts text from an uploaded PDF file."""
    reader = pdf.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_gemini_response(prompt):
    """Gets a response from the Gemini API."""
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

def handle_resume_match(resume_file, job_description):
    """Handles quick resume match requests."""
    try:
        # Extract resume text
        resume_text = extract_pdf_text(resume_file)
        
        # Create prompt for matching percentage
        matching_prompt = f"""
        Hey Act Like a skilled or very experienced ATS. Your task is to evaluate the resume based on the given job description. 
        Only provide the JD Match percentage as a response.

        Resume: {resume_text}
        Job Description: {job_description}

        Provide response in the format:
        {{"JD Match": "%"}}
        """

        response = get_gemini_response(matching_prompt)
        response = response.split(':')[1].strip('}% ')
        response = response[1:-2]
        return jsonify({"similarity": int(response)})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal server error"}), 500

def handle_detailed_match(resume_file, job_description):
    """Handles detailed resume analysis requests."""
    try:
        # Extract resume text
        resume_text = extract_pdf_text(resume_file)

        # Create prompt for detailed feedback
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
        Job Description: {job_description}

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

        response = get_gemini_response(feedback_prompt)
        feedback = json.loads(response)
        return jsonify(feedback)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal server error"}), 500