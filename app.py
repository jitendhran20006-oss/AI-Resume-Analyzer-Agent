import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================
# GEMINI CONFIGURATION
# =========================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)


# =========================
# FASTAPI
# =========================

app = FastAPI(
    title="AI Resume Analyzer Pro",
    version="2.0.0"
)


class AnalyzeRequest(BaseModel):
    resume: str
    job_description: str


# =========================
# AI RESPONSE
# =========================

def get_response_text(response):
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result = []

        for item in content:
            if isinstance(item, dict):
                if item.get("text"):
                    result.append(str(item["text"]))
            elif isinstance(item, str):
                result.append(item)
            else:
                text = getattr(item, "text", None)
                if text:
                    result.append(str(text))

        if result:
            return "\n".join(result)

    return str(content)


def analyze_resume(resume, job_description):

    prompt = f"""
You are a professional AI Resume Analyzer.

Compare the resume with the job description.

Rules:
- Use only information provided by the user.
- Never invent skills, experience, education, projects or certificates.
- Be factual and professional.
- Give useful suggestions.
- Keep the report concise.

Return EXACTLY these sections:

OVERALL MATCH:
MATCH_PERCENTAGE: <number from 0 to 100>
Give a short explanation.

MATCHED SKILLS:
- skill

MISSING SKILLS:
- skill

RELEVANT EXPERIENCE:
- item

AI RECOMMENDATIONS:
- recommendation

ATS KEYWORDS:
- keyword


RESUME:
{resume}


JOB DESCRIPTION:
{job_description}
"""

    try:
        response = llm.invoke(prompt)
        return get_response_text(response)

    except Exception as e:
        return "AI analysis failed: " + str(e)


# =========================
# FRONTEND
# =========================

HTML = """
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>AI Resume Analyzer Pro</title>

<style>

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: Arial, Helvetica, sans-serif;
    background: #f5f7fb;
    color: #172033;
    line-height: 1.6;
}


/* NAVBAR */

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
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: #111827;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
}

.logo span {
    color: #2563eb;
}

.badge {
    background: #eef4ff;
    color: #2563eb;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: bold;
}


/* HERO */

.hero {
    max-width: 1100px;
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
    max-width: 700px;
    margin: auto;
    color: #64748b;
    font-size: 17px;
}


/* MAIN */

.container {
    max-width: 1100px;
    margin: auto;
    padding: 10px 25px 60px;
}

.input-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 22px;
}


/* CARDS */

.card,
.result-card,
.full-report,
.score-card {
    background: white;
    border: 1px solid #e5e9f0;
    border-radius: 16px;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.04);
}

.card {
    padding: 24px;
}

.card-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 15px;
}

.card-title {
    font-size: 17px;
    font-weight: bold;
}

.card-subtitle {
    font-size: 13px;
    color: #94a3b8;
}


/* TEXTAREA */

textarea {
    width: 100%;
    height: 310px;
    resize: vertical;
    border: 1px solid #dce2ea;
    border-radius: 12px;
    padding: 15px;
    font-family: Arial, sans-serif;
    font-size: 14px;
    outline: none;
    color: #334155;
    background: #fafbfc;
}

textarea:focus {
    border-color: #2563eb;
    background: white;
}


/* BUTTONS */

.actions {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin: 25px 0 35px;
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
    min-width: 180px;
}

.primary:hover {
    background: #1d4ed8;
}

.secondary {
    background: white;
    color: #475569;
    border: 1px solid #dce2ea;
}


/* LOADING */

.loading {
    display: none;
    text-align: center;
    margin: 20px;
    color: #64748b;
}

.spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #dbe5ff;
    border-top: 3px solid #2563eb;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    vertical-align: middle;
    margin-right: 8px;
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}


/* RESULTS */

.results {
    display: none;
}

.results-title {
    font-size: 24px;
    margin-bottom: 20px;
}

.score-card {
    padding: 25px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 25px;
}

.score-circle {
    width: 95px;
    height: 95px;
    border-radius: 50%;
    border: 8px solid #2563eb;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    font-weight: bold;
    color: #2563eb;
    flex-shrink: 0;
}

.score-info h3 {
    margin-bottom: 5px;
}

.score-info p {
    color: #64748b;
}


/* RESULT GRID */

.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
}

.result-card {
    padding: 24px;
}

.result-card h3 {
    font-size: 17px;
    margin-bottom: 15px;
}

.result-card ul {
    padding-left: 20px;
}

.result-card li {
    margin-bottom: 8px;
    color: #475569;
}


/* KEYWORDS */

.keyword-card {
    padding: 24px;
    margin-bottom: 20px;
}

.keyword-list {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.keyword {
    background: #eef4ff;
    color: #2563eb;
    padding: 6px 11px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: bold;
}


/* FULL REPORT */

.full-report {
    padding: 25px;
}

.full-report h3 {
    margin-bottom: 15px;
}

.report-text {
    white-space: pre-wrap;
    color: #475569;
    font-size: 14px;
    line-height: 1.8;
}


/* FOOTER */

.footer {
    text-align: center;
    padding: 30px;
    color: #94a3b8;
    font-size: 13px;
}


/* MOBILE */

@media (max-width: 800px) {

    .input-grid,
    .result-grid {
        grid-template-columns: 1fr;
    }

    .hero h1 {
        font-size: 36px;
    }

    .score-card {
        flex-direction: column;
        text-align: center;
    }

}

</style>

</head>


<body>


<!-- NAVBAR -->

<nav class="navbar">

    <div class="logo">

        <div class="logo-icon">
            AI
        </div>

        Resume Analyzer
        <span>Pro</span>

    </div>

    <div class="badge">
        AI Powered
    </div>

</nav>


<!-- HERO -->

<section class="hero">

    <h1>
        Make your resume
        <span>job-ready.</span>
    </h1>

    <p>
        Compare your resume with a target job description,
        discover missing skills, improve your content and
        identify important ATS keywords using AI.
    </p>

</section>


<!-- MAIN -->

<main class="container">


    <!-- INPUTS -->

    <div class="input-grid">


        <!-- RESUME -->

        <div class="card">

            <div class="card-header">

                <div class="card-title">
                    Your Resume
                </div>

                <div class="card-subtitle">
                    Paste resume text
                </div>

            </div>

            <textarea
                id="resume"
                placeholder="Paste your resume here..."
            ></textarea>

        </div>


        <!-- JOB -->

        <div class="card">

            <div class="card-header">

                <div class="card-title">
                    Target Job
                </div>

                <div class="card-subtitle">
                    Paste job description
                </div>

            </div>

            <textarea
                id="job"
                placeholder="Paste the job description here..."
            ></textarea>

        </div>

    </div>


    <!-- BUTTONS -->

    <div class="actions">

        <button
            class="primary"
            onclick="analyze()"
        >
            Analyze Resume
        </button>

        <button
            class="secondary"
            onclick="clearAll()"
        >
            Clear
        </button>

    </div>


    <!-- LOADING -->

    <div
        class="loading"
        id="loading"
    >

        <span class="spinner"></span>

        AI is analyzing your resume...

    </div>


    <!-- RESULTS -->

    <section
        class="results"
        id="results"
    >

        <h2 class="results-title">
            Analysis Results
        </h2>


        <!-- SCORE -->

        <div class="score-card">

            <div
                class="score-circle"
                id="score"
            >
                --
            </div>

            <div class="score-info">

                <h3>
                    Overall Resume Match
                </h3>

                <p id="summary">
                    Analysis completed.
                </p>

            </div>

        </div>


        <!-- RESULT CARDS -->

        <div class="result-grid">


            <div class="result-card">

                <h3>
                    ✓ Matched Skills
                </h3>

                <ul id="matched"></ul>

            </div>


            <div class="result-card">

                <h3>
                    ⚠ Missing Skills
                </h3>

                <ul id="missing"></ul>

            </div>


            <div class="result-card">

                <h3>
                    Relevant Experience
                </h3>

                <ul id="experience"></ul>

            </div>


            <div class="result-card">

                <h3>
                    AI Recommendations
                </h3>

                <ul id="recommendations"></ul>

            </div>

        </div>


        <!-- ATS KEYWORDS -->

        <div class="result-card keyword-card">

            <h3>
                ATS Keywords
            </h3>

            <div
                class="keyword-list"
                id="keywords"
            ></div>

        </div>


        <!-- FULL REPORT -->

        <div class="full-report">

            <h3>
                Complete AI Report
            </h3>

            <div
                class="report-text"
                id="fullReport"
            ></div>

        </div>

    </section>

</main>


<!-- FOOTER -->

<footer class="footer">

    AI Resume Analyzer Pro · Built with Python, FastAPI and Gemini

</footer>


<script>


// =========================
// ANALYZE
// =========================

async function analyze() {

    const resume =
        document.getElementById("resume").value.trim();

    const job =
        document.getElementById("job").value.trim();

    const loading =
        document.getElementById("loading");

    const results =
        document.getElementById("results");


    if (!resume || !job) {

        alert(
            "Please provide both your resume and the job description."
        );

        return;
    }


    loading.style.display = "block";

    results.style.display = "none";


    try {

        const response = await fetch(
            "/analyze",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    resume: resume,
                    job_description: job
                })
            }
        );


        const data = await response.json();


        displayResults(
            data.result || data.detail || "No result."
        );


    } catch (error) {

        alert(
            "Something went wrong. Please try again."
        );

        console.error(error);

    } finally {

        loading.style.display = "none";

    }

}


// =========================
// SECTION PARSER
// =========================

function getSection(text, start, end) {

    const upper =
        text.toUpperCase();

    const startIndex =
        upper.indexOf(
            start.toUpperCase()
        );


    if (startIndex === -1) {
        return "";
    }


    let content =
        text.substring(
            startIndex + start.length
        );


    if (end) {

        const endIndex =
            content.toUpperCase().indexOf(
                end.toUpperCase()
            );


        if (endIndex !== -1) {

            content =
                content.substring(
                    0,
                    endIndex
                );
        }
    }


    return content.trim();

}


// =========================
// CLEAN TEXT
// =========================

function cleanText(text) {

    return text
        .replace(
            /MATCH_PERCENTAGE:\\s*\\d+/gi,
            ""
        )
        .replace(
            /^[-*•]\\s*/gm,
            ""
        )
        .trim();

}


// =========================
// FILL LIST
// =========================

function fillList(id, text) {

    const element =
        document.getElementById(id);

    element.innerHTML = "";


    const lines =
        text
            .split("\\n")
            .map(
                line =>
                    line
                        .replace(
                            /^[-*•]\\s*/,
                            ""
                        )
                        .trim()
            )
            .filter(
                line =>
                    line.length > 0
            );


    if (lines.length === 0) {

        const li =
            document.createElement("li");

        li.textContent =
            "No specific items identified.";

        element.appendChild(li);

        return;
    }


    lines.forEach(
        line => {

            const li =
                document.createElement("li");

            li.textContent = line;

            element.appendChild(li);

        }
    );

}


// =========================
// KEYWORDS
// =========================

function fillKeywords(text) {

    const container =
        document.getElementById("keywords");

    container.innerHTML = "";


    const lines =
        text
            .split("\\n")
            .map(
                line =>
                    line
                        .replace(
                            /^[-*•]\\s*/,
                            ""
                        )
                        .trim()
            )
            .filter(
                line =>
                    line.length > 0
            );


    lines.forEach(
        keyword => {

            const span =
                document.createElement("span");

            span.className =
                "keyword";

            span.textContent =
                keyword;

            container.appendChild(span);

        }
    );

}


// =========================
// DISPLAY RESULTS
// =========================

function displayResults(report) {


    document.getElementById(
        "fullReport"
    ).textContent = report;


    const match =
        report.match(
            /MATCH_PERCENTAGE:\\s*(\\d+)/i
        );


    let percentage = 0;


    if (match) {

        percentage =
            Math.min(
                100,
                Math.max(
                    0,
                    parseInt(match[1])
                )
            );

    }


    document.getElementById(
        "score"
    ).textContent =
        percentage + "%";


    const overall =
        getSection(
            report,
            "OVERALL MATCH:",
            "MATCHED SKILLS:"
        );


    document.getElementById(
        "summary"
    ).textContent =
        cleanText(overall);


    fillList(
        "matched",
        getSection(
            report,
            "MATCHED SKILLS:",
            "MISSING SKILLS:"
        )
    );


    fillList(
        "missing",
        getSection(
            report,
            "MISSING SKILLS:",
            "RELEVANT EXPERIENCE:"
        )
    );


    fillList(
        "experience",
        getSection(
            report,
            "RELEVANT EXPERIENCE:",
            "AI RECOMMENDATIONS:"
        )
    );


    fillList(
        "recommendations",
        getSection(
            report,
            "AI RECOMMENDATIONS:",
            "ATS KEYWORDS:"
        )
    );


    fillKeywords(
        getSection(
            report,
            "ATS KEYWORDS:",
            null
        )
    );


    document.getElementById(
        "results"
    ).style.display = "block";


    document.getElementById(
        "results"
    ).scrollIntoView({
        behavior: "smooth"
    });

}


// =========================
// CLEAR
// =========================

function clearAll() {

    document.getElementById(
        "resume"
    ).value = "";

    document.getElementById(
        "job"
    ).value = "";

    document.getElementById(
        "results"
    ).style.display = "none";

    document.getElementById(
        "score"
    ).textContent = "--";

}

</script>


</body>

</html>
"""


# =========================
# HOME PAGE
# =========================

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


# =========================
# ANALYZE API
# =========================

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


# =========================
# START SERVER
# =========================

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
