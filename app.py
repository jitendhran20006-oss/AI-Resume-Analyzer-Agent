import os
import time
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# GEMINI API CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Career Agent",
    description="AI-powered career assistant for students and job seekers",
    version="1.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class CareerRequest(BaseModel):
    resume: str = ""
    job_description: str = ""
    text: str = ""
    tool: str


# ============================================================
# AI FUNCTION WITH RETRY
# ============================================================

def ask_ai(prompt: str):

    max_attempts = 3

    for attempt in range(max_attempts):

        try:
            response = llm.invoke(prompt)

            return response.content

        except Exception as e:

            error_message = str(e)

            # Retry temporary Gemini 503 errors
            if "503" in error_message or "UNAVAILABLE" in error_message:

                if attempt < max_attempts - 1:
                    time.sleep(3)
                    continue

                return (
                    "Gemini AI is temporarily unavailable because the model "
                    "is experiencing high demand. Please wait a few seconds "
                    "and try again."
                )

            return f"AI service error: {error_message}"


# ============================================================
# CAREER AGENT
# ============================================================

def career_agent(request: CareerRequest):

    tool = request.tool

    # --------------------------------------------------------
    # RESUME ANALYZER
    # --------------------------------------------------------

    if tool == "resume":

        prompt = f"""
You are an expert AI resume analyzer.

Analyze the following resume.

RESUME:
{request.resume}

Provide:

1. Overall Resume Summary
2. Strengths
3. Weaknesses
4. Technical Skills
5. Missing Skills
6. Projects Analysis
7. Education Analysis
8. ATS Improvement Suggestions
9. Specific Resume Improvements
10. Final Career Advice

Keep the answer clear and useful for a B.Tech student.
"""

    # --------------------------------------------------------
    # JOB MATCH
    # --------------------------------------------------------

    elif tool == "job":

        prompt = f"""
You are an AI job matching assistant.

Compare the candidate resume with the job description.

RESUME:
{request.resume}

JOB DESCRIPTION:
{request.job_description}

Provide:

1. Matching Skills
2. Missing Skills
3. Relevant Projects
4. Education Match
5. Technical Skill Match
6. Important Missing Requirements
7. Suggestions to Improve the Resume for This Job
8. Interview Preparation Topics

Do not invent information.
"""

    # --------------------------------------------------------
    # SKILL GAP
    # --------------------------------------------------------

    elif tool == "skills":

        prompt = f"""
You are an AI career skill-gap analyzer.

Analyze the candidate information below.

RESUME:
{request.resume}

Identify:

1. Current Skills
2. Strong Skills
3. Missing Skills
4. Skills Required for AI/ML Jobs
5. Priority Skills to Learn
6. Beginner Level Skills
7. Intermediate Level Skills
8. Advanced Level Skills
9. Recommended Learning Order

Create a practical roadmap for a B.Tech AI/ML student.
"""

    # --------------------------------------------------------
    # RESUME IMPROVER
    # --------------------------------------------------------

    elif tool == "improve":

        prompt = f"""
You are an expert professional resume writer.

Improve the following resume.

RESUME:
{request.resume}

Rewrite it in a professional ATS-friendly format.

Include:

1. Professional Summary
2. Technical Skills
3. Projects
4. Education
5. Achievements if available
6. Better action words
7. ATS-friendly wording

Do not invent qualifications, experience or achievements.
Only improve the information provided.
"""

    # --------------------------------------------------------
    # COVER LETTER
    # --------------------------------------------------------

    elif tool == "cover":

        prompt = f"""
You are an expert career assistant.

Create a professional cover letter using the information below.

RESUME:
{request.resume}

JOB DESCRIPTION:
{request.job_description}

The cover letter should:

- Be professional
- Be suitable for an internship/job application
- Highlight relevant skills
- Mention relevant projects
- Be concise
- Avoid making up experience
"""

    # --------------------------------------------------------
    # PROJECT IDEAS
    # --------------------------------------------------------

    elif tool == "projects":

        prompt = f"""
You are an AI/ML project mentor.

Based on the candidate information below:

RESUME:
{request.resume}

Suggest 8 strong AI/ML software projects.

For every project provide:

1. Project Name
2. Problem it solves
3. Main Features
4. Technologies
5. AI/ML concepts
6. Difficulty Level
7. Why it is useful for a resume

Prefer practical projects that can be deployed online.
"""

    # --------------------------------------------------------
    # LEARNING ROADMAP
    # --------------------------------------------------------

    elif tool == "roadmap":

        prompt = f"""
You are an AI/ML career mentor.

Create a personalized learning roadmap based on:

RESUME:
{request.resume}

Create:

1. Current Level
2. Python Roadmap
3. Data Science Roadmap
4. Machine Learning Roadmap
5. Deep Learning Roadmap
6. Generative AI Roadmap
7. AI Agent Roadmap
8. GitHub Roadmap
9. Project Roadmap
10. Internship Preparation
11. Interview Preparation

Make the roadmap practical and suitable for a B.Tech student.
"""

    # --------------------------------------------------------
    # MOCK INTERVIEW
    # --------------------------------------------------------

    elif tool == "interview":

        prompt = f"""
You are an AI technical interviewer.

Candidate Resume:

{request.resume}

Create a mock interview containing:

1. HR Questions
2. Python Questions
3. Machine Learning Questions
4. AI Questions
5. Project Questions
6. Git/GitHub Questions
7. Generative AI Questions

Ask questions one at a time.

Start with the first interview question.
"""

    # --------------------------------------------------------
    # CAREER CHAT
    # --------------------------------------------------------

    elif tool == "chat":

        prompt = f"""
You are an AI Career Assistant.

Answer the user's career-related question.

USER QUESTION:
{request.text}

Give a clear, practical and beginner-friendly answer.

Focus on:
- AI
- Machine Learning
- Python
- Data Science
- Software Development
- GitHub
- Resume
- Internships
- Jobs
- Career preparation
"""

    else:

        return "Invalid tool selected."


    return ask_ai(prompt)


# ============================================================
# API ENDPOINT
# ============================================================

@app.post("/career")
def career(request: CareerRequest):

    result = career_agent(request)

    return {
        "result": result
    }


# ============================================================
# FRONTEND
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>AI Career Agent</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family: Arial, sans-serif;

    background: #f4f7fb;

    color: #1f2937;

}

.navbar {

    background: #111827;

    color: white;

    padding: 18px 40px;

    display: flex;

    justify-content: space-between;

    align-items: center;

}

.logo {

    font-size: 22px;

    font-weight: bold;

}

.navbar span {

    color: #9ca3af;

    font-size: 14px;

}

.container {

    max-width: 1200px;

    margin: 40px auto;

    padding: 0 20px;

}

.hero {

    text-align: center;

    margin-bottom: 35px;

}

.hero h1 {

    font-size: 38px;

    margin-bottom: 10px;

}

.hero p {

    color: #6b7280;

    font-size: 17px;

}

.tools {

    display: grid;

    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));

    gap: 18px;

    margin-bottom: 30px;

}

.tool {

    background: white;

    padding: 22px;

    border-radius: 14px;

    border: 2px solid transparent;

    cursor: pointer;

    transition: 0.2s;

    box-shadow: 0 4px 15px rgba(0,0,0,0.05);

}

.tool:hover {

    transform: translateY(-3px);

}

.tool.active {

    border-color: #2563eb;

}

.tool h3 {

    margin: 8px 0;

}

.tool p {

    font-size: 13px;

    color: #6b7280;

}

.form-box {

    background: white;

    padding: 28px;

    border-radius: 16px;

    box-shadow: 0 4px 20px rgba(0,0,0,0.06);

}

label {

    display: block;

    font-weight: bold;

    margin-top: 15px;

    margin-bottom: 8px;

}

textarea {

    width: 100%;

    min-height: 150px;

    padding: 14px;

    border: 1px solid #d1d5db;

    border-radius: 10px;

    resize: vertical;

    font-family: Arial;

}

button {

    margin-top: 20px;

    padding: 13px 25px;

    border: none;

    border-radius: 10px;

    background: #2563eb;

    color: white;

    font-size: 16px;

    cursor: pointer;

}

button:hover {

    background: #1d4ed8;

}

.clear {

    background: #6b7280;

    margin-left: 8px;

}

.result {

    margin-top: 30px;

    background: white;

    padding: 28px;

    border-radius: 16px;

    box-shadow: 0 4px 20px rgba(0,0,0,0.06);

}

.result pre {

    white-space: pre-wrap;

    font-family: Arial;

    line-height: 1.7;

}

.loading {

    display: none;

    margin-top: 20px;

    color: #2563eb;

    font-weight: bold;

}

</style>

</head>


<body>


<div class="navbar">

    <div class="logo">🤖 AI Career Agent</div>

    <span>AI-powered career assistant</span>

</div>


<div class="container">


<div class="hero">

    <h1>Build Your AI Career 🚀</h1>

    <p>

        Analyze your resume, match jobs, find skill gaps,

        improve your career profile and prepare for interviews.

    </p>

</div>


<div class="tools">


<div class="tool active" onclick="selectTool('resume', this)">

    📄

    <h3>Resume Analyzer</h3>

    <p>Analyze your resume and get improvement suggestions.</p>

</div>


<div class="tool" onclick="selectTool('job', this)">

    🎯

    <h3>Job Match</h3>

    <p>Compare your resume with a job description.</p>

</div>


<div class="tool" onclick="selectTool('skills', this)">

    🛠️

    <h3>Skill Gap</h3>

    <p>Find missing skills required for your career.</p>

</div>


<div class="tool" onclick="selectTool('improve', this)">

    ✍️

    <h3>Resume Improver</h3>

    <p>Improve your resume using ATS-friendly language.</p>

</div>


<div class="tool" onclick="selectTool('cover', this)">

    📝

    <h3>Cover Letter</h3>

    <p>Create a professional job application cover letter.</p>

</div>


<div class="tool" onclick="selectTool('projects', this)">

    💡

    <h3>Project Ideas</h3>

    <p>Get practical AI/ML project recommendations.</p>

</div>


<div class="tool" onclick="selectTool('roadmap', this)">

    📚

    <h3>Learning Roadmap</h3>

    <p>Get a personalized AI/ML learning roadmap.</p>

</div>


<div class="tool" onclick="selectTool('interview', this)">

    🎤

    <h3>Mock Interview</h3>

    <p>Practice technical and HR interview questions.</p>

</div>


<div class="tool" onclick="selectTool('chat', this)">

    💬

    <h3>Career Chat</h3>

    <p>Ask the AI Career Agent career-related questions.</p>

</div>


</div>


<div class="form-box">


<h2 id="toolTitle">📄 Resume Analyzer</h2>


<label>Resume</label>

<textarea

id="resume"

placeholder="Paste your resume here..."></textarea>


<label id="jobLabel">Job Description</label>

<textarea

id="job"

placeholder="Paste the job description here if required..."></textarea>


<label id="textLabel">Question / Additional Information</label>

<textarea

id="text"

placeholder="Enter your question if required..."></textarea>


<button onclick="runAgent()">

    🚀 Run AI Agent

</button>


<button class="clear" onclick="clearInputs()">

    Clear

</button>


<div class="loading" id="loading">

    🤖 AI is analyzing... Please wait.

</div>


</div>


<div class="result">

<h2>📊 AI Report</h2>

<pre id="result">

Your AI analysis will appear here.

</pre>

</div>


</div>


<script>


let selectedTool = "resume";


function selectTool(tool, element) {


    selectedTool = tool;


    document.querySelectorAll(".tool").forEach(function(card) {

        card.classList.remove("active");

    });


    element.classList.add("active");


    const titles = {

        resume: "📄 Resume Analyzer",

        job: "🎯 Job Match",

        skills: "🛠️ Skill Gap Analyzer",

        improve: "✍️ Resume Improver",

        cover: "📝 Cover Letter Generator",

        projects: "💡 Project Suggestions",

        roadmap: "📚 Learning Roadmap",

        interview: "🎤 Mock Interview",

        chat: "💬 Career Chat"

    };


    document.getElementById("toolTitle").innerText = titles[tool];

}


async function runAgent() {


    const resume = document.getElementById("resume").value;

    const job = document.getElementById("job").value;

    const text = document.getElementById("text").value;


    const loading = document.getElementById("loading");

    const result = document.getElementById("result");


    loading.style.display = "block";

    result.innerText = "AI is working...";


    try {


        const response = await fetch("/career", {

            method: "POST",

            headers: {

                "Content-Type": "application/json"

            },

            body: JSON.stringify({

                resume: resume,

                job_description: job,

                text: text,

                tool: selectedTool

            })

        });


        const data = await response.json();


        result.innerText = data.result;


    }


    catch (error) {


        result.innerText =

            "Something went wrong. Please try again.";


    }


    finally {


        loading.style.display = "none";

    }

}


function clearInputs() {


    document.getElementById("resume").value = "";

    document.getElementById("job").value = "";

    document.getElementById("text").value = "";

    document.getElementById("result").innerText =

        "Your AI analysis will appear here.";

}


</script>


</body>

</html>
"""


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return HTML


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
