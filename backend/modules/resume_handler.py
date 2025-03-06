
import PyPDF2 as pdf
import google.generativeai as genai
import json
from flask import jsonify

# Configure Google AI API
GOOGLE_API_KEY = "AIzaSyCcWfA5IhzK1hZkr5gONwvALAsYN5hOZu8"
genai.configure(api_key=GOOGLE_API_KEY)

# Correct model name
MODEL_NAME = 'gemini-1.5-pro'

def extract_pdf_text(file):
    """Extracts text from an uploaded PDF file."""
    reader = pdf.PdfReader(file)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return text.strip()

def get_gemini_response(prompt):
    """Gets a response from the Gemini API."""
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
            print("⚠️ Gemini AI returned an invalid response:", raw_text)
            return None
    except Exception as e:
        print(f"❌ Gemini API Error: {str(e)}")
        return None

def handle_resume_match(resume_file, job_description):
    """Handles quick resume match requests."""
    try:
        # Extract resume text
        resume_text = extract_pdf_text(resume_file)
        
        # Create prompt for matching percentage
        prompt = f"""
        Act as a professional ATS system. Analyze the resume against the job description and provide ONLY a JSON response.

        Resume: {resume_text}
        Job Description: {job_description}

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
            match_score = response["JD Match"].replace("%", "")
            try:
                return jsonify({"similarity": int(match_score)})
            except ValueError:
                # If we can't convert to int, return as string
                return jsonify({"similarity": match_score})
        else:
            return jsonify({"error": "AI response did not contain a valid match value"}), 400

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal server error"}), 500

def handle_detailed_match(resume_file, job_description):
    """Handles detailed resume analysis requests."""
    try:
        # Extract resume text
        resume_text = extract_pdf_text(resume_file)

        # Create prompt for detailed feedback
        prompt = f"""
        Act as a highly skilled ATS system. Analyze the resume against the job description and return a detailed analysis in JSON format.

        Resume: {resume_text}
        Job Description: {job_description}

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
            return jsonify(response)
        else:
            return jsonify({"error": "AI response did not return a valid JSON object"}), 400

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal server error"}), 500
