# AI Resume Analyzer Agent

An AI-powered resume analysis application built with Python, Gemini, LangChain, FastAPI, and Pydantic.

## Features

- Compare a resume with a target job description
- Identify matched skills
- Identify missing skills
- Analyze relevant experience and projects
- Suggest resume improvements
- Suggest useful job-description keywords
- Simple browser interface
- FastAPI backend
- Gemini-powered analysis

## Technologies

- Python
- Google Gemini
- LangChain
- FastAPI
- Pydantic
- Uvicorn

## Run locally

Install dependencies:

```bash
py -m pip install -r requirements.txt
```

Set your Gemini API key.

Windows CMD:

```cmd
set GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Run:

```bash
py app.py
```

Open:

```text
http://127.0.0.1:8000
```

## Deployment

For Render, create a Python web service from this GitHub repository.

Build Command:

```text
pip install -r requirements.txt
```

Start Command:

```text
python app.py
```

Add the environment variable:

```text
GEMINI_API_KEY
```

Do not put API keys inside the source code.
