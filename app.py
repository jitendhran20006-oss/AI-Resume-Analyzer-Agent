import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")


llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Career Agent",
    description="AI-powered career assistant",
    version="1.0.0"
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
# AI FUNCTION
# ============================================================

def ask_ai(prompt: str) -> str:

    try:

        response = llm.invoke(prompt)

        content = getattr(response, "content", response)

        if isinstance(content, str):
            return content

        if isinstance(content, list):

            parts = []

            for item in content:

                if isinstance(item, dict):

                    if item.get("text"):
                        parts.append(str(item["text"]))

                elif isinstance(item, str):

                    parts.append(item)

                else:

                    text = getattr(item, "text", None)

                    if text:
                        parts.append(str(text))

            if parts:
                return "\n".join(parts)

        return str(content)

    except Exception as e:

        return "AI service error: " + str(e)


# ============================================================
# CAREER AGENT
# ============================================================

def career_agent(request: CareerRequest) -> str:

    resume = request.resume.strip()

    job = request.job_description.strip()

    text = request.text.strip()

    tool = request.tool


    # --------------------------------------------------------
    # RESUME ANALYZER
    # --------------------------------------------------------

    if tool == "resume":

        prompt = f"""
You are a professional AI Resume Analyzer.

Analyze the candidate's resume.

Use ONLY the information provided.

Do not invent qualifications, skills, projects,
experience, certificates or achievements.

Return:

1. RESUME SUMMARY
2. KEY STRENGTHS
3. WEAK AREAS
4. TECHNICAL SKILLS
5. PROJECTS
6. EDUCATION
7. IMPROVEMENT SUGGESTIONS

RESUME:

{resume}
"""


    # --------------------------------------------------------
    # JOB MATCH
    # --------------------------------------------------------

    elif tool == "job":

        prompt = f"""
You are an AI Job Match Analyzer.

Compare the resume with the job description.

Use ONLY information supplied.

Return:

MATCH PERCENTAGE: <number from 0 to 100>

MATCH SUMMARY:

MATCHED SKILLS:
- item

MISSING SKILLS:
- item

MATCHED REQUIREMENTS:
- item

MISSING REQUIREMENTS:
- item

RECOMMENDATIONS:
- item

RESUME:

{resume}

JOB DESCRIPTION:

{job}
"""


    # --------------------------------------------------------
    # SKILL GAP
    # --------------------------------------------------------

    elif tool == "skills":

        prompt = f"""
You are an AI Skill Gap Analyzer.

Analyze the candidate's current skills against the
target job requirements.

Do not invent skills.

Return:

CURRENT SKILLS:
- skill

REQUIRED SKILLS:
- skill

SKILLS TO LEARN:
- skill

PRIORITY SKILLS:
- skill

RECOMMENDED LEARNING ORDER:
1.
2.
3.
4.

RESUME:

{resume}

JOB DESCRIPTION:

{job}
"""


    # --------------------------------------------------------
    # RESUME IMPROVER
    # --------------------------------------------------------

    elif tool == "improve":

        prompt = f"""
You are a professional resume improvement assistant.

Improve the supplied resume for the supplied job description.

Do NOT invent experience or qualifications.

Preserve the candidate's real information.

Return:

PROFESSIONAL SUMMARY:

IMPROVED SKILLS SECTION:

IMPROVED PROJECT DESCRIPTIONS:

IMPROVED EXPERIENCE:

KEY IMPROVEMENTS:

ATS KEYWORDS:

RESUME:

{resume}

JOB DESCRIPTION:

{job}
"""


    # --------------------------------------------------------
    # COVER LETTER
    # --------------------------------------------------------

    elif tool == "cover":

        prompt = f"""
You are a professional cover letter writer.

Create a professional job-specific cover letter.

Use ONLY information available in the resume.

Do not invent experience or qualifications.

Make it suitable for an internship or entry-level position.

Return a polished cover letter.

RESUME:

{resume}

JOB DESCRIPTION:

{job}
"""


    # --------------------------------------------------------
    # PROJECT SUGGESTIONS
    # --------------------------------------------------------

    elif tool == "projects":

        prompt = f"""
You are an AI project advisor for a student looking
for internships and entry-level AI/ML jobs.

Based on the candidate's resume and target job,
suggest useful portfolio projects.

Do not claim the candidate already built projects
unless they appear in the resume.

Suggest 5 projects.

For each project give:

PROJECT NAME:
PURPOSE:
TECHNOLOGIES:
AI/ML CONCEPTS:
KEY FEATURES:
WHY IT HELPS THE CANDIDATE:

RESUME:

{resume}

JOB DESCRIPTION:

{job}
"""


    # --------------------------------------------------------
    # LEARNING ROADMAP
    # --------------------------------------------------------

    elif tool == "roadmap":

        prompt = f"""
You are an AI career learning advisor.

Create a practical learning roadmap based on
the candidate's current skills and target job.

Do not assume skills that are not shown.

Create:

CURRENT LEVEL:

GOAL:

PHASE 1:
Topics:
Projects:

PHASE 2:
Topics:
Projects:

PHASE 3:
Topics:
Projects:

PHASE 4:
Topics:
Projects:

RECOMMENDED TOOLS:

FINAL PORTFOLIO PLAN:

RESUME:

{resume}

JOB DESCRIPTION:

{job}
"""


    # --------------------------------------------------------
    # MOCK INTERVIEW
    # --------------------------------------------------------

    elif tool == "interview":

        prompt = f"""
You are an AI technical interviewer.

Create a mock interview for the candidate.

The candidate is applying for the target job.

Create 10 interview questions.

Include:

- 4 technical questions
- 2 project questions
- 2 behavioral questions
- 2 job-specific questions

For each question provide:

QUESTION:
WHAT THE INTERVIEWER IS TESTING:
WHAT A GOOD ANSWER SHOULD INCLUDE:

Do not invent experience for the candidate.

RESUME:

{resume}

JOB DESCRIPTION:

{job}
"""


    # --------------------------------------------------------
    # CAREER CHAT
    # --------------------------------------------------------

    elif tool == "chat":

        prompt = f"""
You are an AI Career Assistant.

Answer the user's career question clearly.

Give practical advice for a college student
interested in AI, ML and software careers.

Do not invent personal information.

USER QUESTION:

{text}
"""


    else:

        return "Please select a valid career tool."


    return ask_ai(prompt)


# ============================================================
# API ENDPOINT
# ============================================================

@app.post("/career")
def career(request: CareerRequest):

    return {
        "result": career_agent(request)
    }


# ============================================================
# FRONTEND
# ============================================================

HTML = """
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>AI Career Agent</title>


<style>

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}


body {

    font-family:
    Arial,
    Helvetica,
    sans-serif;

    background: #f5f7fb;

    color: #172033;

    line-height: 1.6;
}


/* =====================================================
   NAVBAR
   ===================================================== */

.navbar {

    height: 70px;

    background: white;

    border-bottom: 1px solid #e5e7eb;

    display: flex;

    align-items: center;

    justify-content: space-between;

    padding: 0 6%;

}


.logo {

    display: flex;

    align-items: center;

    gap: 10px;

    font-size: 20px;

    font-weight: bold;
}


.logo-icon {

    width: 40px;

    height: 40px;

    background: #111827;

    color: white;

    border-radius: 11px;

    display: flex;

    align-items: center;

    justify-content: center;

    font-weight: bold;
}


.logo span {

    color: #2563eb;
}


.badge {

    background: #eef4ff;

    color: #2563eb;

    padding: 6px 13px;

    border-radius: 20px;

    font-size: 13px;

    font-weight: bold;
}


/* =====================================================
   HERO
   ===================================================== */

.hero {

    max-width: 1050px;

    margin: auto;

    text-align: center;

    padding: 60px 25px 35px;
}


.hero h1 {

    font-size: 46px;

    line-height: 1.15;

    margin-bottom: 18px;
}


.hero h1 span {

    color: #2563eb;
}


.hero p {

    max-width: 720px;

    margin: auto;

    color: #64748b;

    font-size: 17px;
}


/* =====================================================
   MAIN
   ===================================================== */

.container {

    max-width: 1100px;

    margin: auto;

    padding: 10px 25px 60px;
}


/* =====================================================
   TOOLS
   ===================================================== */

.tools {

    display: grid;

    grid-template-columns:
    repeat(4, 1fr);

    gap: 15px;

    margin-bottom: 25px;
}


.tool {

    background: white;

    border: 1px solid #e5e9f0;

    border-radius: 14px;

    padding: 18px;

    cursor: pointer;

    transition: 0.2s;

    text-align: left;
}


.tool:hover {

    border-color: #2563eb;

    transform: translateY(-2px);
}


.tool.active {

    border: 2px solid #2563eb;

    background: #f8fbff;
}


.tool-icon {

    font-size: 25px;

    margin-bottom: 8px;
}


.tool-title {

    font-weight: bold;

    font-size: 15px;
}


.tool-description {

    color: #64748b;

    font-size: 12px;

    margin-top: 4px;
}


/* =====================================================
   INPUT CARD
   ===================================================== */

.input-card {

    background: white;

    border: 1px solid #e5e9f0;

    border-radius: 16px;

    padding: 25px;

    box-shadow:
    0 8px 25px
    rgba(15, 23, 42, 0.04);
}


.input-title {

    font-size: 19px;

    font-weight: bold;

    margin-bottom: 6px;
}


.input-subtitle {

    color: #64748b;

    font-size: 14px;

    margin-bottom: 18px;
}


textarea {

    width: 100%;

    height: 220px;

    resize: vertical;

    border:
    1px solid #dce2ea;

    border-radius: 12px;

    padding: 15px;

    font-family: Arial;

    font-size: 14px;

    outline: none;

    background: #fafbfc;
}


textarea:focus {

    border-color: #2563eb;

    background: white;
}


.second-input {

    display: none;

    margin-top: 18px;
}


.second-input label {

    display: block;

    font-weight: bold;

    margin-bottom: 8px;
}


.action-row {

    display: flex;

    justify-content: center;

    gap: 12px;

    margin-top: 20px;
}


button {

    border: none;

    border-radius: 10px;

    padding: 13px 25px;

    font-size: 15px;

    font-weight: bold;

    cursor: pointer;
}


.primary {

    background: #2563eb;

    color: white;

    min-width: 190px;
}


.primary:hover {

    background: #1d4ed8;
}


.secondary {

    background: white;

    color: #475569;

    border: 1px solid #dce2ea;
}


/* =====================================================
   LOADING
   ===================================================== */

.loading {

    display: none;

    text-align: center;

    padding: 25px;

    color: #64748b;
}


.spinner {

    display: inline-block;

    width: 20px;

    height: 20px;

    border:
    3px solid #dbe5ff;

    border-top:
    3px solid #2563eb;

    border-radius: 50%;

    animation:
    spin 0.8s linear infinite;

    vertical-align: middle;

    margin-right: 8px;
}


@keyframes spin {

    to {
        transform: rotate(360deg);
    }

}


/* =====================================================
   RESULT
   ===================================================== */

.result {

    display: none;

    margin-top: 25px;
}


.result-header {

    background: white;

    border:
    1px solid #e5e9f0;

    border-radius: 16px;

    padding: 25px;

    margin-bottom: 20px;
}


.result-header h2 {

    margin-bottom: 5px;
}


.result-header p {

    color: #64748b;

    font-size: 14px;
}


.report {

    background: white;

    border:
    1px solid #e5e9f0;

    border-radius: 16px;

    padding: 28px;

    white-space: pre-wrap;

    color: #334155;

    line-height: 1.8;

    font-size: 14px;

    box-shadow:
    0 8px 25px
    rgba(15, 23, 42, 0.04);
}


/* =====================================================
   FOOTER
   ===================================================== */

.footer {

    text-align: center;

    padding: 35px;

    color: #94a3b8;

    font-size: 13px;
}


/* =====================================================
   RESPONSIVE
   ===================================================== */

@media(max-width: 900px) {

    .tools {

        grid-template-columns:
        repeat(2, 1fr);
    }

}


@media(max-width: 600px) {

    .tools {

        grid-template-columns: 1fr;
    }

    .hero h1 {

        font-size: 35px;
    }

    .navbar {

        padding: 0 20px;
    }

}

</style>

</head>


<body>


<!-- =====================================================
     NAVBAR
     ===================================================== -->

<nav class="navbar">

    <div class="logo">

        <div class="logo-icon">
            AI
        </div>

        AI Career
        <span>Agent</span>

    </div>


    <div class="badge">
        AI Powered
    </div>

</nav>


<!-- =====================================================
     HERO
     ===================================================== -->

<section class="hero">

    <h1>

        Your Personal
        <span>AI Career Agent</span>

    </h1>


    <p>

        Analyze your resume, match jobs,
        discover skill gaps, improve your resume,
        prepare for interviews and build your
        personalized career roadmap.

    </p>

</section>


<!-- =====================================================
     MAIN
     ===================================================== -->

<main class="container">


<!-- TOOLS -->

<div class="tools">


    <div
        class="tool active"
        data-tool="resume"
        onclick="selectTool('resume', this)"
    >

        <div class="tool-icon">
            📄
        </div>

        <div class="tool-title">
            Resume Analyzer
        </div>

        <div class="tool-description">
            Analyze your resume
        </div>

    </div>


    <div
        class="tool"
        data-tool="job"
        onclick="selectTool('job', this)"
    >

        <div class="tool-icon">
            🎯
        </div>

        <div class="tool-title">
            Job Match
        </div>

        <div class="tool-description">
            Compare resume with jobs
        </div>

    </div>


    <div
        class="tool"
        data-tool="skills"
        onclick="selectTool('skills', this)"
    >

        <div class="tool-icon">
            🛠️
        </div>

        <div class="tool-title">
            Skill Gap
        </div>

        <div class="tool-description">
            Find skills to learn
        </div>

    </div>


    <div
        class="tool"
        data-tool="improve"
        onclick="selectTool('improve', this)"
    >

        <div class="tool-icon">
            ✍️
        </div>

        <div class="tool-title">
            Resume Improver
        </div>

        <div class="tool-description">
            Improve your resume
        </div>

    </div>


    <div
        class="tool"
        data-tool="cover"
        onclick="selectTool('cover', this)"
    >

        <div class="tool-icon">
            📝
        </div>

        <div class="tool-title">
            Cover Letter
        </div>

        <div class="tool-description">
            Generate a cover letter
        </div>

    </div>


    <div
        class="tool"
        data-tool="projects"
        onclick="selectTool('projects', this)"
    >

        <div class="tool-icon">
            💡
        </div>

        <div class="tool-title">
            Project Ideas
        </div>

        <div class="tool-description">
            Get portfolio projects
        </div>

    </div>


    <div
        class="tool"
        data-tool="roadmap"
        onclick="selectTool('roadmap', this)"
    >

        <div class="tool-icon">
            📚
        </div>

        <div class="tool-title">
            Learning Roadmap
        </div>

        <div class="tool-description">
            Build your career plan
        </div>

    </div>


    <div
        class="tool"
        data-tool="interview"
        onclick="selectTool('interview', this)"
    >

        <div class="tool-icon">
            🎤
        </div>

        <div class="tool-title">
            Mock Interview
        </div>

        <div class="tool-description">
            Practice interviews
        </div>

    </div>


</div>


<!-- INPUT -->

<div class="input-card">


    <div class="input-title"
         id="inputTitle">

        Resume Analyzer

    </div>


    <div class="input-subtitle"
         id="inputSubtitle">

        Paste your resume below and let AI
        analyze your career profile.

    </div>


    <textarea
        id="mainText"
        placeholder="Paste your resume here..."
    ></textarea>


    <!-- SECOND INPUT -->

    <div
        class="second-input"
        id="secondInput"
    >

        <label id="secondLabel">

            Job Description

        </label>


        <textarea
            id="secondText"
            placeholder="Paste the job description here..."
        ></textarea>

    </div>


    <!-- BUTTONS -->

    <div class="action-row">

        <button
            class="primary"
            onclick="runAgent()"
        >

            Run AI Agent

        </button>


        <button
            class="secondary"
            onclick="clearInputs()"
        >

            Clear

        </button>

    </div>


</div>


<!-- LOADING -->

<div
    class="loading"
    id="loading"
>

    <span class="spinner"></span>

    AI Career Agent is working...

</div>


<!-- RESULT -->

<section
    class="result"
    id="result"
>


    <div class="result-header">

        <h2>
            AI Career Report
        </h2>

        <p>
            Generated by your AI Career Agent
        </p>

    </div>


    <div
        class="report"
        id="report"
    ></div>


</section>


</main>


<!-- FOOTER -->

<footer class="footer">

    AI Career Agent · Built with Python,
    FastAPI, LangChain and Gemini

</footer>


<script>


// ========================================================
// CURRENT TOOL
// ========================================================

let currentTool = "resume";


// ========================================================
// TOOL INFORMATION
// ========================================================

const toolInfo = {

    resume: {

        title: "Resume Analyzer",

        subtitle:
        "Paste your resume below and let AI analyze your career profile.",

        placeholder:
        "Paste your resume here...",

        second: false

    },


    job: {

        title: "Job Match Analyzer",

        subtitle:
        "Compare your resume with a target job description.",

        placeholder:
        "Paste your resume here...",

        second: true

    },


    skills: {

        title: "Skill Gap Analyzer",

        subtitle:
        "Discover the skills you need to learn for your target job.",

        placeholder:
        "Paste your resume and current skills here...",

        second: true

    },


    improve: {

        title: "AI Resume Improver",

        subtitle:
        "Improve your resume while keeping your real experience.",

        placeholder:
        "Paste your current resume here...",

        second: true

    },


    cover: {

        title: "Cover Letter Generator",

        subtitle:
        "Generate a professional job-specific cover letter.",

        placeholder:
        "Paste your resume here...",

        second: true

    },


    projects: {

        title: "AI Project Advisor",

        subtitle:
        "Get portfolio project ideas based on your career target.",

        placeholder:
        "Paste your resume and current skills here...",

        second: true

    },


    roadmap: {

        title: "Learning Roadmap",

        subtitle:
        "Build a personalized learning path for your career goal.",

        placeholder:
        "Paste your resume and current skills here...",

        second: true

    },


    interview: {

        title: "AI Mock Interview",

        subtitle:
        "Generate a personalized technical and behavioral interview.",

        placeholder:
        "Paste your resume here...",

        second: true

    }

};


// ========================================================
// SELECT TOOL
// ========================================================

function selectTool(tool, element) {

    currentTool = tool;


    document
        .querySelectorAll(".tool")
        .forEach(
            item =>
                item.classList.remove("active")
        );


    element.classList.add("active");


    const info =
        toolInfo[tool];


    document.getElementById(
        "inputTitle"
    ).textContent =
        info.title;


    document.getElementById(
        "inputSubtitle"
    ).textContent =
        info.subtitle;


    document.getElementById(
        "mainText"
    ).placeholder =
        info.placeholder;


    const second =
        document.getElementById(
            "secondInput"
        );


    if (info.second) {

        second.style.display =
            "block";

    } else {

        second.style.display =
            "none";

    }


    document.getElementById(
        "result"
    ).style.display =
        "none";

}


// ========================================================
// RUN AGENT
// ========================================================

async function runAgent() {

    const mainText =
        document.getElementById(
            "mainText"
        ).value.trim();


    const secondText =
        document.getElementById(
            "secondText"
        ).value.trim();


    if (!mainText) {

        alert(
            "Please enter the required information."
        );

        return;

    }


    if (
        toolInfo[currentTool].second
        &&
        !secondText
    ) {

        alert(
            "Please provide the second input."
        );

        return;

    }


    const loading =
        document.getElementById(
            "loading"
        );


    const result =
        document.getElementById(
            "result"
        );


    loading.style.display =
        "block";


    result.style.display =
        "none";


    try {

        const response =
            await fetch(
                "/career",
                {

                    method: "POST",

                    headers: {

                        "Content-Type":
                        "application/json"

                    },

                    body:
                    JSON.stringify({

                        tool:
                        currentTool,

                        resume:
                        currentTool === "chat"
                        ? ""
                        : mainText,

                        job_description:
                        toolInfo[currentTool].second
                        ? secondText
                        : "",

                        text:
                        mainText

                    })

                }
            );


        const data =
            await response.json();


        document.getElementById(
            "report"
        ).textContent =
            data.result ||
            "No result returned.";


        result.style.display =
            "block";


        result.scrollIntoView({
            behavior: "smooth"
        });


    } catch (error) {

        document.getElementById(
            "report"
        ).textContent =
            "Something went wrong: "
            + error.message;


        result.style.display =
            "block";

    } finally {

        loading.style.display =
            "none";

    }

}


// ========================================================
// CLEAR
// ========================================================

function clearInputs() {

    document.getElementById(
        "mainText"
    ).value = "";


    document.getElementById(
        "secondText"
    ).value = "";


    document.getElementById(
        "result"
    ).style.display =
        "none";

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

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
