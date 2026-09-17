import os
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# GEMINI API KEY
# =========================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")


# =========================================================
# GEMINI MODEL
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="AI Resume Analyzer Agent",
    description="AI-powered resume and job description analyzer",
    version="1.0.0"
)


# =========================================================
# REQUEST MODEL
# =========================================================

class AnalyzeRequest(BaseModel):
    resume: str
    job_description: str


# =========================================================
# EXTRACT GEMINI RESPONSE
# =========================================================

def extract_response_text(response):

    content = getattr(response, "content", response)

    # Normal string response
    if isinstance(content, str):
        return content

    # Gemini sometimes returns a list of content blocks
    if isinstance(content, list):

        result = []

        for item in content:

            if isinstance(item, dict):

                if "text" in item:
                    result.append(str(item["text"]))

                elif "content" in item:
                    result.append(str(item["content"]))

            elif isinstance(item, str):
                result.append(item)

            else:
                # Try to get text attribute
                text = getattr(item, "text", None)

                if text:
                    result.append(str(text))

        if result:
            return "\n".join(result)

    return str(content)


# =========================================================
# RESUME ANALYZER
# =========================================================

def analyze_resume(resume, job_description):

    prompt = f"""
You are an AI Resume Analyzer Agent.

Your task is to compare the candidate's resume with the target job description.

Analyze ONLY the information provided.

Do NOT invent skills, experience, education, certificates, or achievements.

Return the answer in a clear professional format.

Use exactly these sections:

1. OVERALL MATCH

Give a short explanation of how well the resume matches the job.

2. MATCHED SKILLS

List the skills present in both the resume and job description.

3. MISSING SKILLS

List important skills mentioned in the job description that are not clearly present in the resume.

4. RELEVANT PROJECTS AND EXPERIENCE

Explain which projects or experience from the resume are relevant to the job.

5. RESUME IMPROVEMENT SUGGESTIONS

Give practical suggestions to improve the resume for this job.

6. SUGGESTED KEYWORDS

List keywords from the job description that could naturally be included in the resume if they are genuinely applicable.

--------------------------------------------------

RESUME:

{resume}

--------------------------------------------------

JOB DESCRIPTION:

{job_description}

--------------------------------------------------
"""

    try:

        response = llm.invoke(prompt)

        return extract_response_text(response)

    except Exception as e:

        return f"AI analysis failed: {str(e)}"


# =========================================================
# HTML FRONTEND
# =========================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>AI Resume Analyzer Agent</title>

<style>

* {
    box-sizing: border-box;
}

body {

    font-family: Arial, sans-serif;

    background: #f4f6f8;

    margin: 0;

    padding: 30px;

}

.container {

    max-width: 1000px;

    margin: auto;

    background: white;

    padding: 30px;

    border-radius: 12px;

    box-shadow: 0 4px 20px rgba(0,0,0,0.08);

}

h1 {

    text-align: center;

    margin-bottom: 10px;

}

.subtitle {

    text-align: center;

    color: #666;

    margin-bottom: 30px;

}

label {

    display: block;

    font-weight: bold;

    margin-top: 20px;

    margin-bottom: 8px;

}

textarea {

    width: 100%;

    min-height: 220px;

    padding: 14px;

    border: 1px solid #ccc;

    border-radius: 8px;

    font-size: 15px;

    resize: vertical;

}

button {

    display: block;

    margin: 25px auto;

    padding: 14px 30px;

    border: none;

    border-radius: 8px;

    background: #222;

    color: white;

    font-size: 16px;

    cursor: pointer;

}

button:hover {

    background: #444;

}

button:disabled {

    background: #999;

    cursor: not-allowed;

}

#output {

    margin-top: 25px;

    padding: 20px;

    background: #f8f8f8;

    border: 1px solid #ddd;

    border-radius: 8px;

    white-space: pre-wrap;

    line-height: 1.6;

    min-height: 100px;

}

.error {

    color: #b00020;

}

</style>

</head>


<body>


<div class="container">

<h1>AI Resume Analyzer Agent</h1>

<p class="subtitle">

Compare your resume with a target job description using AI.

</p>


<label>Resume</label>

<textarea

id="resume"

placeholder="Paste your resume here..."

></textarea>


<label>Job Description</label>

<textarea

id="job"

placeholder="Paste the target job description here..."

></textarea>


<button id="analyzeButton" onclick="analyzeResume()">

Analyze Resume

</button>


<div id="output">

Analysis will appear here.

</div>


</div>


<script>


async function analyzeResume() {

    const resume =
        document.getElementById("resume").value.trim();

    const job =
        document.getElementById("job").value.trim();

    const output =
        document.getElementById("output");

    const button =
        document.getElementById("analyzeButton");


    if (!resume || !job) {

        output.innerHTML =
            '<span class="error">Please provide both resume and job description.</span>';

        return;

    }


    button.disabled = true;

    button.textContent = "Analyzing...";

    output.textContent =
        "AI is analyzing your resume. Please wait...";


    try {

        const response = await fetch("/analyze", {

            method: "POST",

            headers: {

                "Content-Type": "application/json"

            },

            body: JSON.stringify({

                resume: resume,

                job_description: job

            })

        });


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail || "Server error"
            );

        }


        if (typeof data.result === "string") {

            output.textContent = data.result;

        }

        else {

            output.textContent =
                JSON.stringify(data.result, null, 2);

        }


    }

    catch (error) {

        output.innerHTML =
            '<span class="error">Error: ' +
            error.message +
            '</span>';

    }


    button.disabled = false;

    button.textContent = "Analyze Resume";

}


</script>


</body>

</html>
"""


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return HTML


# =========================================================
# ANALYZE API
# =========================================================

@app.post("/analyze")
def analyze(request: AnalyzeRequest):

    if not request.resume.strip():

        return {
            "result": "Please provide your resume."
        }


    if not request.job_description.strip():

        return {
            "result": "Please provide the job description."
        }


    result = analyze_resume(
        request.resume,
        request.job_description
    )


    return {
        "result": result
    }


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
