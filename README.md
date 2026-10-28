# 🎓 AI Study & Career Navigator

**AI Study & Career Navigator** is a university midterm project developed with Python, Streamlit, and Google's official Gemini API (`google-genai` Python SDK). It provides personalized, evidence-based academic and career guidance for university students based on their major, degree level, skills, interests, and aspirations.

The application supports **Google's Gemini API** (using the `gemini-2.5-flash` model with structured Pydantic outputs) and features an **offline Local Career Guidance fallback engine** that operates completely free of charge without requiring any API key or payment.

---

## 🛠️ Technology Stack

- **Language:** Python 3.12+
- **Web Interface:** [Streamlit](https://streamlit.io/)
- **AI Engine & Structured Outputs:** Official [Google GenAI SDK](https://github.com/googleapis/python-genai) (`google-genai`) with Gemini structured schema enforcement
- **Default AI Model:** `gemini-2.5-flash` (fast, structured output support, and free tier available)
- **Fallback Engine:** Offline rule-based career matching engine (deterministic, zero cost, no API key needed)
- **Data Validation & Schemas:** [Pydantic](https://docs.pydantic.dev/) v2
- **Automated Testing:** [pytest](https://docs.pytest.org/) (100% offline mocked unit tests)
- **Version Control:** Git & GitHub

---

## 📂 Project Structure

```text
ai-study-career-navigator/
│
├── .streamlit/
│   ├── secrets.toml.example   # Safe template for Gemini API key configuration
│   └── secrets.toml           # (Local only, ignored by Git) Your Gemini API key
├── tests/
│   └── test_ai_engine.py      # Automated unit tests using pytest and mocks
├── .gitignore                 # Excludes secrets, virtual environments, and caches
├── ai_engine.py               # Gemini integration, Pydantic schemas, and local fallback engine
├── app.py                     # Streamlit frontend with interactive tabs, forms, and guidance modes
├── requirements.txt           # Project dependencies
└── README.md                  # Project overview and setup documentation
```

---

## 🚀 Getting Started Locally

### 1. Open the Project
Open PowerShell or your preferred terminal in the project root:
```bash
cd ai-study-career-navigator
```

### 2. Activate Your Virtual Environment

**On Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If script execution is disabled on your system, enable it for your session: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`, or run `.\.venv\Scripts\streamlit.exe run app.py` directly)*

**On macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 Google Gemini API Configuration & Free Tier Details

### Step 1: Obtain a Free Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google Account.
3. Click **Get API key** and create a new key in a free Google Cloud project.
4. Copy your API key (starts with `AIzaSy...`).

### Step 2: Configure Streamlit Secrets
1. Create a copy of `.streamlit/secrets.toml.example` named `.streamlit/secrets.toml`:
   ```powershell
   copy .streamlit\secrets.toml.example .streamlit\secrets.toml
   ```
2. Open `.streamlit/secrets.toml` in your editor and enter your key:
   ```toml
   GEMINI_API_KEY = "AIzaSy..."
   ```
3. **Security Confirmation:** The file `.streamlit/secrets.toml` is strictly listed in `.gitignore` and is **never** committed or pushed to Git or GitHub.

*(Alternatively, you can test the application by pasting your Gemini API key directly into the sidebar session override field in the web interface.)*

### ⚠️ Understanding Gemini API Free Tier Usage Limits
Google AI Studio offers a free API tier for models like `gemini-2.5-flash`. Please keep the following in mind:
- **Rate Limits:** The free tier includes limits on Requests Per Minute (RPM), Requests Per Day (RPD), and Tokens Per Minute (TPM).
- **Quota Exceeded (HTTP 429):** If you make too many requests within a brief window, Google's servers may temporarily return a rate limit error.
- **Graceful Handling:** If a rate limit is reached or if no API key is provided, the application will notify you and seamlessly allow you to explore career paths using the **Local Career Guidance** mode.

---

## 🧭 Fallback Mode: Offline Local Career Guidance

You do **NOT** need to pay for an OpenAI API key or even configure a Gemini API key to use this application:

- If **no Gemini API key** is configured, the application automatically launches in **Local Career Guidance** mode.
- The local engine uses transparent, rule-based matching across the student's field of study, skills, interests, career goal, and preferred location.
- Results are clearly identified as "Local Career Guidance" and adhere strictly to the same Pydantic `CareerAnalysisReport` schema.
- The engine guarantees exactly 3 tailored career paths, skills-gap analysis, learning resources, and a 6-month roadmap across 3 phases.
- It never invents fake match percentages, fabricated salary figures, or hiring guarantees.

---

## 🧪 Running the Automated Unit Tests

All 27 unit tests run **100% offline** without making live network calls or using API quota:

```powershell
.\.venv\Scripts\pytest.exe -v
```

The test suite covers:
- User profile creation, required field validation, and study level defaults
- Prompt construction and list parsing
- Gemini API key validation and placeholder detection
- Schema constraints (strictly enforcing 3 recommendations and 3 roadmap phases)
- Mocked Gemini responses using `google-genai` structured outputs
- Mocked Gemini authentication errors (HTTP 403 / `PERMISSION_DENIED`)
- Mocked Gemini rate limit & quota errors (HTTP 429 / `RESOURCE_EXHAUSTED`)
- Mocked network and server connection errors (HTTP 503 / `UNAVAILABLE`)
- Offline Local Career Guidance engine (100% offline, zero API keys required)
- Finance domain skill prioritization (verifying advanced Excel, financial modeling, SQL, Power BI, Python, statistics over DevOps/software engineering tools)
- Deduplication and logical next-level skill recommendations
- Realistic progressive 6-month roadmap with applied portfolio projects (financial dashboards, valuation)
- Verifiable learning resource URLs (Coursera, CFI, Microsoft Learn, edX)
- Fallback schema compliance and ethical disclaimers across diverse academic majors

---

## 💻 Launching the Application

Start the Streamlit application:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Open your browser to:
👉 **`http://localhost:8501`**

---

## 🌟 Application Features & Sections

When an analysis is completed (via Gemini AI or Local Guidance), the application presents your results across four organized tabs:

1. **🧭 Recommended Career Pathways:** Exactly three tailored career trajectories with typical responsibilities, transferable skills, and key growth areas.
2. **📊 Skills Gap Analysis:** Compares your existing skills against industry expectations, highlighting strengths, technical gaps, and soft skills to build.
3. **📚 Learning Resources & Certifications:** Curated courses and certifications categorized by provider and topic *(includes an advisory notice reminding students that certifications build competencies, but do not guarantee hiring)*.
4. **🗓️ 6-Month Action Roadmap:** A sequential timeline split into three 2-month phases with milestones and weekly action items.

---

## 📌 Development Status

- [x] Clean project structure & virtual environment setup
- [x] Interactive questionnaire form with field validation
- [x] Pydantic structured output models for career analysis
- [x] Google Gemini API integration with `gemini-2.5-flash` using official `google-genai` SDK
- [x] Streamlit secrets configuration (`GEMINI_API_KEY`) with `.gitignore` protection
- [x] Offline Local Career Guidance fallback engine (zero cost, rule-based)
- [x] Robust error handling (invalid keys, rate limits, network outages)
- [x] Automated test suite (27/27 tests passing offline)
- [ ] Career roadmap export (PDF / Markdown download) (Milestone 3)
