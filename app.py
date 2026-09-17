import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)

app = FastAPI(
    title="AI Resume Analyzer Agent",
    description="An AI agent that analyzes resumes against job descriptions.",
    version="1.0.0"
)

class AnalyzeRequest(BaseModel):
    resume: str = Field(description="Resume text")
    job_description: str = Field(description="Job description")

def analyze_resume(resume: str, job_description: str) -> str:
    prompt = f"""
You are an AI Resume Analyzer Agent.

Analyze the resume against the job description.

RESUME:
{resume}

JOB DESCRIPTION:
{job_description}

Return a clear report with these sections:
1. Overall Match
2. Matched Skills
3. Missing Skills
4. Relevant Experience and Projects
5. Resume Improvement Suggestions
6. Suggested Keywords

Do not invent qualifications or experience. Base the analysis only on the supplied text.
"""
    response = llm.invoke(prompt)
    content = response.content

if isinstance(content, str):
    return content

if isinstance(content, list):
    text = ""
    for item in content:
        if isinstance(item, dict):
            text += item.get("text", "")
        else:
            text += str(item)
    return text

return str(content)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>AI Resume Analyzer Agent</title>
<style>
body{font-family:Arial;max-width:1000px;margin:40px auto;padding:20px}
textarea{width:100%;height:220px;margin:8px 0 20px;padding:12px;box-sizing:border-box}
button{padding:12px 22px;font-size:16px;cursor:pointer}
#output{white-space:pre-wrap;margin-top:25px;padding:20px;border:1px solid #ccc}
</style>
</head>
<body>
<h1>AI Resume Analyzer Agent</h1>
<p>Paste your resume and the target job description below.</p>

<label><b>Resume</b></label>
<textarea id="resume" placeholder="Paste resume text here..."></textarea>

<label><b>Job Description</b></label>
<textarea id="job" placeholder="Paste job description here..."></textarea>

<button onclick="analyze()">Analyze Resume</button>

<div id="output">Analysis will appear here.</div>

<script>
async function analyze(){
    const output=document.getElementById("output");
    output.textContent="Analyzing...";
    const response=await fetch("/analyze",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({
            resume:document.getElementById("resume").value,
            job_description:document.getElementById("job").value
        })
    });
    const data=await response.json();
    output.textContent=data.result || data.detail || "Something went wrong.";
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    if not request.resume.strip() or not request.job_description.strip():
        return {"result": "Please provide both a resume and a job description."}
    return {"result": analyze_resume(request.resume, request.job_description)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
