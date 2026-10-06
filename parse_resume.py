import os

from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from docx import Document


load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key not available")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"

role="user"

#structure the data format
from pydantic import BaseModel
class Resume(BaseModel):
    match_percent: str
    name: str
    email: str
    phone: str
    experience: str

    matched_skills: list[str]
    missing_skills: list[str]

    matching_reasons: list[str]
    missing_requirements: list[str]

    final_verdict: str
    verdict_reason: str

schema = Resume.model_json_schema()

response_format = {
  "type": "json_object"
}

system_prompt = f"""
You are an expert resume screening and job matching system.

Your task is to analyze a candidate's resume against the provided job requirements.

Return ONLY valid JSON.
Do not return markdown.
Do not return explanations outside the JSON.

The JSON must EXACTLY follow this structure:

{{
    "match_percent": "86%",
    "name": "AVINASH KUMAR",
    "email": "example@gmail.com",
    "phone": "9876543210",
    "experience": "5 years",

    "matched_skills": [
        "React.js",
        "JavaScript",
        "TypeScript"
    ],

    "missing_skills": [
        "Docker",
        "Kubernetes"
    ],

    "matching_reasons": [
        "The candidate has 5 years of frontend development experience.",
        "The candidate has strong React.js and TypeScript experience.",
        "The candidate's experience matches the core frontend requirements."
    ],

    "missing_requirements": [
        "The resume does not provide sufficient evidence of Docker experience.",
        "The resume does not mention Kubernetes experience."
    ],

    "final_verdict": "Strong Match",

    "verdict_reason": "The candidate matches most of the important requirements but has some gaps in infrastructure-related skills."
}}

Rules:

1. Extract the candidate's personal information only from the resume.
2. Compare the resume ONLY against the provided job requirements.
3. Do not assume that a skill exists if it is not supported by the resume.
4. Do not invent experience, skills, projects, companies, or technologies.
5. matched_skills must contain skills explicitly supported by the resume and relevant to the job requirements.
6. missing_skills must contain important job requirements that are not present or not sufficiently demonstrated in the resume.
7. matching_reasons must explain WHY the candidate matches the job.
8. missing_requirements must explain WHY certain requirements are considered missing.
9. Match percentage must reflect the overall alignment between the resume and the job requirements.
10. Do not calculate the match percentage based only on the number of skills.
11. Give higher importance to mandatory/core requirements than optional/nice-to-have requirements.
12. Experience level, technical skills, responsibilities, and relevant domain experience should all be considered.
13. If a requirement is not mentioned in the resume, treat it as "not demonstrated", not automatically as "does not have the skill".
14. Do not penalize the candidate heavily for optional or nice-to-have requirements.
15. final_verdict must be one of:
    - "Excellent Match"
    - "Strong Match"
    - "Moderate Match"
    - "Weak Match"
    - "Not a Match"

16. Use these approximate guidelines for final_verdict:
    - 90-100%: Excellent Match
    - 75-89%: Strong Match
    - 60-74%: Moderate Match
    - 40-59%: Weak Match
    - 0-39%: Not a Match

17. match_percent must be a string containing the percentage sign.
18. experience must be a string.
19. phone must be a string.
20. matched_skills must always be an array of strings.
21. missing_skills must always be an array of strings.
22. matching_reasons must always be an array of strings.
23. missing_requirements must always be an array of strings.
24. Do not add any fields other than the fields specified above.
25. Do not use nested objects.
"""

message_system = {
  "role": "system",
  "content": system_prompt
}


# --------------------------------------------------
# PDF
# --------------------------------------------------

def extract_pdf_text(file_path):
    reader = PdfReader(file_path)

    text = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text)


# --------------------------------------------------
# DOCX
# --------------------------------------------------

def extract_docx_text(file_path):
    document = Document(file_path)

    text = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text.append(paragraph.text)

    return "\n".join(text)


# --------------------------------------------------
# TXT
# --------------------------------------------------

def extract_txt_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# --------------------------------------------------
# Common text extractor
# --------------------------------------------------

def extract_text(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf_text(path)

    elif extension == ".docx":
        return extract_docx_text(path)

    elif extension == ".txt":
        return extract_txt_text(path)

    else:
        raise ValueError(
            "Unsupported file type. Supported formats: PDF, DOCX, TXT"
        )


# --------------------------------------------------
# File input
# --------------------------------------------------

resume_path = "./Avinash_Kumar_Resume_Final_2026.pdf"
requirements_path = "./requirements.txt"

resume_text = extract_text(resume_path)
requirement_text = extract_text(requirements_path)


# print("\nExtracted text:")
# print(resume_text)


# --------------------------------------------------
# Send extracted text to LLM
# --------------------------------------------------

prompt = f"""
Analyze the following candidate resume and job requirements.

================ RESUME ================
{resume_text}

================ JOB REQUIREMENTS ================
{requirement_text}

================ TASK ================

Compare the resume against the job requirements carefully.

Identify:

1. Overall match percentage.
2. Candidate information.
3. Skills that are clearly present in the resume AND relevant to the job.
4. Important skills/requirements from the job that are missing or not demonstrated in the resume.
5. Specific reasons why the candidate matches the job.
6. Specific reasons why the candidate does not fully match the job.
7. A final hiring-match verdict.

When evaluating the candidate:

- Prioritize mandatory requirements.
- Consider years of experience.
- Consider technical skills.
- Consider frameworks and libraries.
- Consider responsibilities mentioned in the job description.
- Consider relevant project/domain experience.
- Distinguish between "missing" and "not demonstrated".
- Do not assume skills that are not explicitly supported by the resume.
- Do not give credit for a skill merely because it is related to another skill.
- Do not invent information.

For example:

If the job requires:
"5+ years React and TypeScript"

and the resume says:
"5 years React and TypeScript"

then these should be considered matched.

If the job requires:
"Docker and Kubernetes"

and the resume does not mention them, they should be considered missing/not demonstrated.

Return ONLY the JSON object according to the required schema.
"""


message = {
  "role": role,
  "content": prompt
}

messages = [message_system, message]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    response_format=response_format
)


answer = response.choices[0].message.content


print("\nLLM ANSWER:")
print(answer)


import json
data = json.loads(answer)
resume = Resume.model_validate(data)

print(resume)
