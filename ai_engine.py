"""
AI Engine Module - AI Study & Career Navigator
----------------------------------------------
This module handles communication with Google's Gemini API using the official
google-genai Python SDK with structured output validation (Pydantic schema),
generating tailored study and career guidance based on student input.

It also provides an offline, deterministic Local Career Guidance engine that
operates when no Gemini API key is configured, guaranteeing that students can
always explore realistic career pathways, personalized skills gaps, and roadmaps
without requiring paid API access or external network calls.
"""

import os
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
import httpx

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
# Default Gemini model with a generous free API tier in Google AI Studio.
DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"
# Retained as a backward-compatibility alias for legacy imports
DEFAULT_OPENAI_MODEL: str = DEFAULT_GEMINI_MODEL

__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "CAREER_ADVISOR_SYSTEM_INSTRUCTIONS",
    "UserProfile",
    "CareerRecommendation",
    "SkillsGapAnalysis",
    "LearningResource",
    "RoadmapPhase",
    "CareerAnalysisReport",
    "CareerGuidanceError",
    "MissingAPIKeyError",
    "AuthenticationAPIError",
    "RateLimitAPIError",
    "NetworkAPIError",
    "InvalidResponseError",
    "parse_comma_separated_list",
    "validate_student_input",
    "validate_api_key",
    "build_recommendation_prompt",
    "generate_career_guidance",
    "generate_local_guidance",
]

CAREER_ADVISOR_SYSTEM_INSTRUCTIONS: str = (
    "You are an expert, compassionate university career counselor and academic mentor. "
    "Your mission is to provide realistic, actionable career pathways, a structured skills gap "
    "analysis, high-quality learning resources with verifiable URLs, and a sequential 6-month roadmap "
    "based on the student's major, degree level, skills, interests, and aspirations.\n\n"
    "CRITICAL ETHICAL & DOMAIN GUIDELINES:\n"
    "1. Never invent fake statistical match percentages (e.g., '95% match' or 'Score: 88%').\n"
    "2. Never guarantee employment, job offers, or hiring outcomes. Make it clear that certifications "
    "and courses build competencies and demonstrate initiative, but do not promise jobs.\n"
    "3. Provide exactly three (3) distinct, realistic career recommendations suited for their background.\n"
    "4. TAILOR SKILLS GAPS TO THE TARGET DOMAIN: For financial and business roles (e.g. Financial Data Analyst, "
    "FP&A, Investment Analyst), prioritize domain-appropriate competencies: advanced Excel, financial modeling, "
    "SQL, Power BI/Tableau, Python with pandas, applied statistics, and business communication. DO NOT recommend "
    "software-engineering infrastructure tools (Docker, Kubernetes, CI/CD, Git branches) as default priorities for finance careers.\n"
    "5. DEDUPLICATE SKILLS: Never list a skill in technical_skills_to_acquire if the student already lists it as a current skill. "
    "Instead, recommend advanced next-level competencies (e.g., if they know Excel, suggest advanced dynamic financial modeling; "
    "if they know SQL, suggest window functions and optimization).\n"
    "6. VERIFIABLE RESOURCES: Provide real, well-known learning resources with genuine platform URLs (e.g. Coursera, edX, Microsoft Learn, CFI). "
    "Do not invent fake courses or dead URLs.\n"
    "7. Provide a progressive 6-month roadmap organized into three 2-month phases (Months 1-2, Months 3-4, Months 5-6) "
    "centered around a concrete portfolio project (e.g., an end-to-end financial dashboard or valuation analysis for finance roles).\n"
    "8. Ensure advice is practical, encouraging, and tailored to university students and recent grads."
)


# -----------------------------------------------------------------------------
# Custom Exception Hierarchy
# -----------------------------------------------------------------------------
class CareerGuidanceError(Exception):
    """Base exception for all AI Career Guidance failures."""
    pass


class MissingAPIKeyError(CareerGuidanceError):
    """Raised when the Gemini API key is missing, empty, or a placeholder."""
    pass


class AuthenticationAPIError(CareerGuidanceError):
    """Raised when the Gemini API key is invalid or unauthorized."""
    pass


class RateLimitAPIError(CareerGuidanceError):
    """Raised when the Gemini rate limit or quota is exceeded."""
    pass


class NetworkAPIError(CareerGuidanceError):
    """Raised when network connectivity to Gemini fails or times out."""
    pass


class InvalidResponseError(CareerGuidanceError):
    """Raised when the AI response cannot be parsed into the expected report schema."""
    pass


# -----------------------------------------------------------------------------
# Student Input Models
# -----------------------------------------------------------------------------
class UserProfile(BaseModel):
    """
    Data model representing a student's profile and preferences.
    Using Pydantic ensures data validation and type safety.
    """
    academic_background: str = Field(..., min_length=2, description="Major, degree, or current field of study")
    study_level: str = Field(default="Undergraduate", description="Current study level (e.g., Undergraduate, Master's, Doctoral)")
    skills: str = Field(..., min_length=2, description="Current skills (technical and soft skills)")
    interests: str = Field(..., min_length=2, description="Academic and industry interests")
    goals: str = Field(..., min_length=2, description="Short-term or long-term career goals")
    preferred_location: str = Field(..., min_length=2, description="Preferred working location or remote preference")


# -----------------------------------------------------------------------------
# Response Schema Models (Pydantic Structured Outputs)
# -----------------------------------------------------------------------------
class CareerRecommendation(BaseModel):
    """A tailored career pathway recommendation."""
    role_title: str = Field(..., description="Professional title or role name")
    rationale: str = Field(..., description="Why this role aligns with the student's background and interests")
    typical_responsibilities: List[str] = Field(..., description="Key everyday responsibilities in this role")
    transferable_skills: List[str] = Field(..., description="Skills the student already possesses that directly apply")
    growth_areas: List[str] = Field(..., description="Specific new skills the student must cultivate for this role")


class SkillsGapAnalysis(BaseModel):
    """Comparative analysis of existing capabilities vs. industry expectations."""
    identified_strengths: List[str] = Field(..., description="Key strengths identified from the student profile")
    technical_skills_to_acquire: List[str] = Field(..., description="Crucial technical or domain skills to learn")
    soft_skills_recommendations: List[str] = Field(..., description="Interpersonal, communication, or leadership competencies")


class LearningResource(BaseModel):
    """Curated courses, certifications, or self-directed project suggestions."""
    name: str = Field(..., description="Title of the course, certification, or project")
    resource_type: str = Field(..., description="Category: Certification, Online Course, Practical Project, or Book")
    provider_or_platform: str = Field(..., description="Issuing organization or host (e.g., Coursera, AWS, edX, GitHub)")
    practical_value: str = Field(..., description="How this resource bridges the student's skill gap")
    url: Optional[str] = Field(default=None, description="Direct verifiable URL to the learning resource or platform")


class RoadmapPhase(BaseModel):
    """A 2-month phase within the six-month career preparation roadmap."""
    month_range: str = Field(..., description="Timeframe covered (e.g., 'Months 1-2', 'Months 3-4', 'Months 5-6')")
    phase_title: str = Field(..., description="High-level focus area for this phase")
    focus_milestones: List[str] = Field(..., description="Measurable milestones to accomplish")
    action_items: List[str] = Field(..., description="Concrete, weekly actionable steps")


class CareerAnalysisReport(BaseModel):
    """
    Comprehensive career guidance report generated by Gemini API or Local Guidance engine.
    Guarantees exactly three recommendations and a six-month roadmap across 3 phases.
    """
    executive_summary: str = Field(..., description="A personalized, encouraging synthesis of the student's profile")
    career_recommendations: List[CareerRecommendation] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Exactly three tailored career recommendations"
    )
    skills_gap: SkillsGapAnalysis = Field(..., description="Detailed skills gap analysis")
    learning_resources: List[LearningResource] = Field(..., description="Recommended certifications and study materials")
    six_month_roadmap: List[RoadmapPhase] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Sequential roadmap covering all six months in three phases"
    )
    advisory_note: str = Field(
        ...,
        description="Ethical disclaimer noting that career paths require effort and certifications do not guarantee hiring"
    )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def parse_comma_separated_list(text: str) -> List[str]:
    """
    Parses a comma-separated string into a clean list of unique items.
    Trims extra whitespace and ignores empty entries.
    """
    if not text:
        return []

    items: List[str] = []
    seen = set()

    for item in text.split(","):
        cleaned = item.strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            items.append(cleaned)

    return items


def validate_student_input(major: str, career_goal: str) -> Tuple[bool, Optional[str]]:
    """
    Validates required fields from the student profile form.
    """
    if not major or len(major.strip()) < 2:
        return False, "Please enter your academic major or field of study (at least 2 characters)."

    if not career_goal or len(career_goal.strip()) < 2:
        return False, "Please describe your career goal or target role (at least 2 characters)."

    return True, None


def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Validates whether a Gemini API key is provided, non-empty, and not a placeholder.
    """
    if not api_key:
        return False
    stripped = api_key.strip()
    placeholders = {
        "your-gemini-api-key-here",
        "your-openai-api-key-here",
        "your-api-key-here",
    }
    return len(stripped) > 0 and stripped.lower() not in placeholders


def build_recommendation_prompt(profile: UserProfile) -> str:
    """
    Constructs a structured prompt for the Gemini model based on the student's profile.
    """
    return (
        "Please analyze the following university student's profile and generate a comprehensive, "
        "personalized CareerAnalysisReport following the required structured schema:\n\n"
        f"- Academic Background: {profile.academic_background}\n"
        f"- Current Study Level: {profile.study_level}\n"
        f"- Current Skills: {profile.skills}\n"
        f"- Interests & Passions: {profile.interests}\n"
        f"- Target Career Goal: {profile.goals}\n"
        f"- Preferred Work Location / Environment: {profile.preferred_location}\n\n"
        "Domain-Specific Personalization Guidelines:\n"
        "1. Prioritize technical skills strictly relevant to their target career domain (e.g., for finance and financial "
        "data analysis roles, prioritize advanced Excel, financial modeling, SQL, Power BI/Tableau, Python with pandas, "
        "and applied statistics; DO NOT recommend software-engineering DevOps tools like Docker or CI/CD).\n"
        "2. Do not recommend skills they already list as current competencies; instead suggest logical next-level mastery.\n"
        "3. Provide real, verifiable URLs for all recommended learning resources.\n"
        "4. Structure the 6-month roadmap around a progressive, practical portfolio project tailored to their field "
        "(e.g., a financial dashboard or valuation analysis for finance roles).\n"
        "Remember to provide exactly 3 career paths and a 6-month roadmap across 3 phases."
    )


# -----------------------------------------------------------------------------
# Core AI Generation Function (Google GenAI SDK)
# -----------------------------------------------------------------------------
def generate_career_guidance(
    profile: UserProfile,
    api_key: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL
) -> CareerAnalysisReport:
    """
    Generates personalized career recommendations using Google's official
    google-genai Python SDK with structured JSON output mapped to CareerAnalysisReport.

    Args:
        profile (UserProfile): The student's academic profile and preferences.
        api_key (Optional[str]): Gemini API key. If omitted, checks GEMINI_API_KEY env var.
        model (str): Gemini model alias to use (defaults to DEFAULT_GEMINI_MODEL).

    Returns:
        CareerAnalysisReport: Validated Pydantic report containing 3 career options,
                             skills gap analysis, resources, and a 6-month roadmap.

    Raises:
        MissingAPIKeyError: If API key is missing or blank.
        AuthenticationAPIError: If Gemini rejects the key (401/403 Unauthorized/Forbidden).
        RateLimitAPIError: If rate limits or quota are exceeded (429 Resource Exhausted).
        NetworkAPIError: If connection fails or times out.
        InvalidResponseError: If response fails Pydantic validation.
        CareerGuidanceError: For any other unexpected Gemini error.
    """
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY")

    if not validate_api_key(resolved_key):
        raise MissingAPIKeyError(
            "Gemini API key is missing or not configured. "
            "Please configure your GEMINI_API_KEY in .streamlit/secrets.toml or provide it in the app."
        )

    try:
        client = genai.Client(api_key=resolved_key)

        config = types.GenerateContentConfig(
            system_instruction=CAREER_ADVISOR_SYSTEM_INSTRUCTIONS,
            response_mime_type="application/json",
            response_schema=CareerAnalysisReport,
        )

        response = client.models.generate_content(
            model=model,
            contents=build_recommendation_prompt(profile),
            config=config,
        )

        # Parse response into CareerAnalysisReport
        if hasattr(response, "parsed") and isinstance(response.parsed, CareerAnalysisReport):
            return response.parsed
        elif hasattr(response, "parsed") and response.parsed is not None:
            if isinstance(response.parsed, dict):
                return CareerAnalysisReport.model_validate(response.parsed)
            return response.parsed
        elif hasattr(response, "output_parsed") and response.output_parsed is not None:
            return response.output_parsed
        elif hasattr(response, "text") and response.text:
            return CareerAnalysisReport.model_validate_json(response.text)
        else:
            raise InvalidResponseError("The Gemini model returned an empty or unparseable response structure.")

    except (ClientError, APIError) as exc:
        status_code = getattr(exc, "code", None)
        err_str = str(exc).lower()

        # 401 Unauthorized / 403 Forbidden / API_KEY_INVALID / PERMISSION_DENIED
        if (
            status_code in (400, 401, 403)
            and ("api key" in err_str or "permission_denied" in err_str or "unauthenticated" in err_str or "invalid" in err_str)
        ) or "api key not valid" in err_str or "permission_denied" in err_str:
            raise AuthenticationAPIError(
                "Authentication failed: The provided Gemini API key is invalid or unauthorized. "
                "Please check your GEMINI_API_KEY in .streamlit/secrets.toml."
            ) from exc

        # 429 RateLimit / Quota Exhaustion / RESOURCE_EXHAUSTED
        if status_code == 429 or "resource_exhausted" in err_str or "quota" in err_str or "rate limit" in err_str:
            raise RateLimitAPIError(
                "Gemini API rate limit or quota exceeded. "
                "The Gemini API free tier has usage limits (e.g., requests per minute and per day). "
                "Please wait a moment before trying again or use Local Career Guidance."
            ) from exc

        # 5xx Server Errors
        if status_code is not None and status_code >= 500:
            raise NetworkAPIError(
                "Could not connect to Google Gemini servers. Please check your internet connection and try again."
            ) from exc

        raise CareerGuidanceError(f"Gemini service error: {exc}") from exc

    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
        raise NetworkAPIError(
            "Could not connect to Google Gemini servers. Please check your internet connection and try again."
        ) from exc

    except ValidationError as exc:
        raise InvalidResponseError(
            f"The Gemini output did not match the expected report schema: {exc}"
        ) from exc

    except Exception as exc:
        if isinstance(exc, CareerGuidanceError):
            raise
        raise CareerGuidanceError(f"Unexpected error during career guidance generation: {exc}") from exc


# -----------------------------------------------------------------------------
# Local Fallback Career Guidance Engine (Deterministic Rule-Based Matching)
# -----------------------------------------------------------------------------
def generate_local_guidance(profile: UserProfile) -> CareerAnalysisReport:
    """
    Generates personalized, tailored career guidance using an offline, deterministic
    rule-based matching engine that evaluates the student's field of study, skills,
    interests, target goal, study level, and preferred location without making any external API calls.

    Personalization Principles:
    - Prioritizes domain-appropriate skills (e.g. advanced Excel, financial modeling, SQL, Power BI,
      Python with pandas, and statistics for financial data analysis).
    - Checks user's existing skills: never recommends an existing skill as a basic gap,
      suggesting a logical next-level competency instead.
    - Keeps software-engineering tools (Git, Docker, CI/CD) as optional supplementary tools,
      never default priorities for finance careers.
    - Structures a realistic, progressive 6-month roadmap with an applied portfolio project.
    - Curates learning resources with useful, verifiable URLs.
    - Transparently labeled as Local Career Guidance.
    """
    major = profile.academic_background.strip()
    study_level = profile.study_level.strip()
    target_goal = profile.goals.strip()
    preferred_loc = profile.preferred_location.strip() or "Flexible / Open location"

    skills_items = parse_comma_separated_list(profile.skills)
    interests_items = parse_comma_separated_list(profile.interests)

    major_lower = major.lower()
    goal_lower = target_goal.lower()
    interests_lower = profile.interests.lower()
    combined_text = f"{major} {target_goal} {profile.skills} {profile.interests}".lower()

    # User existing skill checker
    user_skills_lower = [s.lower().strip() for s in skills_items]

    def has_user_skill(keywords: List[str]) -> bool:
        """Returns True if any keyword is present in any of the user's listed skills."""
        for s in user_skills_lower:
            for kw in keywords:
                if kw in s or s in kw:
                    return True
        return False

    def get_transferable(default_skills: List[str], count: int = 2) -> List[str]:
        """Selects applicable skills from the user's input, or supplies sensible domain defaults."""
        if skills_items:
            return skills_items[:count]
        return default_skills[:count]

    # Domain keyword patterns
    finance_keywords = [
        "finance", "financial", "fintech", "banking", "investment", "accounting",
        "management science", "economics", "actuarial", "commerce", "fp&a",
        "portfolio", "equity", "hedge fund", "asset management", "wealth", "treasury"
    ]
    data_keywords = [
        "data", "analytics", "analyst", "business intelligence", "bi", "quantitative", "quant"
    ]

    is_finance_discipline = (
        any(k in major_lower for k in finance_keywords)
        or any(k in goal_lower for k in finance_keywords)
        or any(k in interests_lower for k in finance_keywords)
    )
    is_data_focused = (
        any(k in goal_lower for k in data_keywords)
        or any(k in major_lower for k in data_keywords)
    )

    # 1. Financial Data Analysis / Quantitative Finance & Business Analytics
    is_financial_data = is_finance_discipline and (
        any(k in goal_lower for k in ["data analyst", "financial data", "financial analyst", "quantitative", "quant", "bi", "intelligence", "analytics", "risk analyst", "fp&a"])
        or is_data_focused
    )

    # 2. General Corporate Finance & Business (non-data-specialized)
    is_finance_general = is_finance_discipline and not is_financial_data

    # 3. Pure Tech AI & Data Science
    is_pure_tech_ai = (
        any(k in combined_text for k in ["machine learning", "deep learning", "artificial intelligence", "nlp", "computer vision", "data scientist", "data engineer"])
        and not is_finance_discipline
    )

    # 4. Pure Software Engineering / Systems / Cloud
    is_software_tech = (
        any(k in combined_text for k in ["computer science", "software", "code", "programming", "backend", "frontend", "full stack", "fullstack", "devops", "cloud", "cyber", "web developer", "systems engineer"])
        and not is_finance_discipline
    )

    # 5. Life Sciences & Healthcare
    is_health_bio = any(k in combined_text for k in ["biology", "biomedical", "biotech", "chemistry", "medicine", "pre-med", "clinical", "genetics", "health", "pharmacy", "neuroscience", "nursing", "biochemistry"])

    # 6. Engineering (Physical / Hardware)
    is_engineering = any(k in combined_text for k in ["mechanical", "electrical", "civil", "aerospace", "hardware", "mechatronics", "industrial engineering", "structural", "cad", "robotics"]) and not is_software_tech

    # 7. Design & UI/UX
    is_design = any(k in combined_text for k in ["design", "ux", "ui", "product design", "graphic", "media", "animation", "creative", "interaction design"])

    # 8. Psychology & Social Sciences
    is_psych_social = any(k in combined_text for k in ["psychology", "sociology", "social work", "human resources", "hr", "counseling", "behavioral", "political science", "public policy", "international relations"])

    # -------------------------------------------------------------------------
    # DOMAIN BRANCH A: Financial Data Analysis
    # -------------------------------------------------------------------------
    if is_financial_data:
        # Check user's current finance skills
        has_excel = has_user_skill(["excel", "spreadsheet", "spreadsheets", "vba"])
        has_fin_model = has_user_skill(["financial modeling", "financial model", "dcf", "valuation", "3-statement"])
        has_sql = has_user_skill(["sql", "mysql", "postgres", "postgresql", "database", "querying"])
        has_bi = has_user_skill(["power bi", "tableau", "bi", "dashboard", "data visualization", "looker"])
        has_python = has_user_skill(["python", "pandas", "numpy", "jupyter"])
        has_stats = has_user_skill(["statistics", "stats", "econometrics", "regression", "quantitative"])

        # Prioritize unpossessed skills first; otherwise suggest logical next level
        candidate_gaps: List[Tuple[bool, str]] = []

        if not has_fin_model:
            candidate_gaps.append((True, "Financial Modeling & Valuation: Building dynamic 3-statement models, DCF valuation, and scenario sensitivity tables"))
        else:
            candidate_gaps.append((False, "Advanced Financial Modeling (Next Level): M&A accretion/dilution analysis, capital structure stress-testing, and Monte Carlo risk simulations"))

        if not has_excel:
            candidate_gaps.append((True, "Advanced Excel for Finance: XLOOKUP/Index-Match, nested financial formulas, dynamic arrays, and Power Query data transformation"))
        else:
            candidate_gaps.append((False, "Advanced Excel Automation (Next Level): Power Query data consolidation, dynamic financial simulation arrays, and automated auditing macros"))

        if not has_sql:
            candidate_gaps.append((True, "Relational Database Querying (SQL): Writing multi-table joins, aggregations, window functions, and CTEs to extract financial ledger data"))
        else:
            candidate_gaps.append((False, "Advanced SQL for Finance (Next Level): Query optimization, recursive CTEs, and automated ETL pipelines for accounting ledgers"))

        if not has_bi:
            candidate_gaps.append((True, "Business Intelligence & Data Visualization (Power BI or Tableau): Building interactive financial KPI dashboards, DAX formulas, and cash flow visualizations"))
        else:
            candidate_gaps.append((False, "Enterprise BI Architecture (Next Level): Advanced DAX measures for financial ratios, automated data model refreshes, and executive KPI reporting"))

        if not has_python:
            candidate_gaps.append((True, "Python for Financial Analysis (pandas & numpy): Automating financial statement data extraction, time-series manipulation, and exploratory data analysis"))
        else:
            candidate_gaps.append((False, "Quantitative Python (Next Level): Econometric modeling (statsmodels), automated SEC EDGAR/financial API ingestion, and portfolio risk backtesting"))

        if not has_stats:
            candidate_gaps.append((True, "Applied Financial Statistics: Regression analysis, hypothesis testing, and time-series variance analysis for financial forecasting"))
        else:
            candidate_gaps.append((False, "Advanced Financial Econometrics (Next Level): Multivariate time-series forecasting (ARIMA/GARCH), Value-at-Risk (VaR), and multi-factor regression"))

        # Sort so skills the user does NOT possess appear first, then next-level skills
        candidate_gaps.sort(key=lambda x: not x[0])
        technical_gaps = [item[1] for item in candidate_gaps[:3]]

        # Soft skills tailored to financial presentation
        soft_skills = [
            "Executive data storytelling: Translating complex financial data, variance metrics, and modeling assumptions into clear strategic takeaways for leadership",
            "Cross-functional collaboration: Partnering between accounting, operational units, and corporate strategy to audit and validate reporting numbers",
            f"Professional financial networking: Engaging alumni, CFA societies, and financial analyst peer groups in {preferred_loc}"
        ]

        # Recommendation 1: Direct Target Career Pathway
        rec1_title = target_goal if any(w in target_goal.lower() for w in ["analyst", "specialist", "associate", "modeler"]) else f"{target_goal} Specialist"
        rec1 = CareerRecommendation(
            role_title=rec1_title,
            rationale=(
                f"Directly advances your target career ambition as a {target_goal}. "
                f"Your coursework in {major} ({study_level}) provides theoretical grounding in financial systems and quantitative reasoning. "
                f"Combining your interest in {profile.interests} with data manipulation prepares you well for data-driven finance teams in {preferred_loc}."
            ),
            typical_responsibilities=[
                "Query relational databases using SQL to extract, clean, and validate transactional and accounting data",
                "Develop and maintain automated financial KPI dashboards tracking revenue, burn rate, and operational margins",
                "Conduct historical variance analysis and build forecasting models to support quarterly financial planning",
                "Collaborate with finance leaders to translate complex analytical findings into executive-ready presentations"
            ],
            transferable_skills=get_transferable(["Financial Statement Analysis", "Quantitative Reasoning", "Excel Spreadsheet Analysis"], 3),
            growth_areas=[
                "Advanced financial modeling (DCF valuation, 3-statement forecasting, and scenario sensitivity tables)",
                "Business intelligence dashboard engineering (Power BI or Tableau with DAX formulas)",
                "Optional Supplementary Tool: Basic version control (Git) for managing financial scripts and analytical documentation (secondary skill)"
            ]
        )

        # Recommendation 2: Adjacent Analytical Track
        rec2 = CareerRecommendation(
            role_title="Business Intelligence Analyst — Finance & Strategy",
            rationale=(
                f"An adjacent, high-demand analytical career track that pairs your {major} background with business intelligence infrastructure. "
                f"Rather than focusing solely on accounting records, this role analyzes cross-functional drivers—such as customer acquisition costs, operational efficiency, and pricing elasticity—for organizations in {preferred_loc}."
            ),
            typical_responsibilities=[
                "Design centralized BI dashboards in Power BI or Tableau linking financial ledgers to departmental operational metrics",
                "Perform deep-dive exploratory data analysis to uncover cost-saving and revenue-expansion opportunities",
                "Partner with data engineering and accounting teams to ensure single-source-of-truth reporting data pipelines",
                "Present monthly KPI executive decks to operational vice presidents and division heads"
            ],
            transferable_skills=get_transferable(["Business Acumen", "Data Visualization", "SQL Database Fundamentals"], 2),
            growth_areas=[
                "Relational database schema design and dimensional modeling (star schema)",
                "Advanced data storytelling and translating quantitative insights into commercial strategy",
                "Automated data extract and refresh scheduling in enterprise BI environments"
            ]
        )

        # Recommendation 3: Strategic Corporate Finance Track
        rec3 = CareerRecommendation(
            role_title="Corporate Financial Planning & Analysis (FP&A) Analyst",
            rationale=(
                f"A core corporate finance pathway that leverages your {major} education to drive budgeting, forecasting, and resource allocation. "
                f"FP&A teams are strategic partners to executive leadership, providing high visibility and strong long-term career growth in {preferred_loc}."
            ),
            typical_responsibilities=[
                "Partner with departmental leaders to construct annual operating budgets and monthly rolling forecasts",
                "Analyze budget-versus-actual variances and author concise qualitative commentary for executive reviews",
                "Evaluate capital expenditure proposals, unit economics, and new business venture feasibility",
                "Maintain and refine long-range integrated corporate financial planning models"
            ],
            transferable_skills=get_transferable(["Budget Variance Analysis", "Financial Accounting", "Spreadsheet Modeling"], 2),
            growth_areas=[
                "Integrated 3-statement financial modeling and balance sheet forecasting",
                "Executive presentation and synthesizing financial drivers for non-financial department heads",
                "Corporate ERP and EPM software systems (e.g., SAP, Oracle NetSuite, or Adaptive Planning)"
            ]
        )

        # Verifiable, real learning resources for financial data analysis
        resources = [
            LearningResource(
                name="Financial Modeling & Valuation Analyst (FMVA®)",
                resource_type="Certification",
                provider_or_platform="Corporate Finance Institute (CFI)",
                practical_value="Comprehensive professional training in dynamic 3-statement financial modeling, DCF valuation, and advanced Excel spreadsheet mechanics (demonstrates initiative; not a job guarantee).",
                url="https://corporatefinanceinstitute.com/certifications/financial-modeling-valuation-analyst-fmva/"
            ),
            LearningResource(
                name="Google Data Analytics Professional Certificate",
                resource_type="Certification",
                provider_or_platform="Coursera / Google",
                practical_value="Hands-on foundation in SQL database querying, Tableau data visualization, and structured problem-solving for business analytics.",
                url="https://www.coursera.org/professional-certificates/google-data-analytics"
            ),
            LearningResource(
                name="Business and Financial Modeling Specialization",
                resource_type="Online Course",
                provider_or_platform="Coursera / Wharton School",
                practical_value="Rigorous academic curriculum covering quantitative spreadsheet modeling, risk scenarios, and decision tree analysis on corporate financial data.",
                url="https://www.coursera.org/specializations/wharton-business-financial-modeling"
            )
        ]

        # Progressive 6-month roadmap with financial dashboard / valuation project
        roadmap = [
            RoadmapPhase(
                month_range="Months 1-2",
                phase_title="Financial Modeling Foundations & SQL Data Mastery",
                focus_milestones=[
                    f"Audit academic coursework in {major} against entry-level Financial Data Analyst requirements in {preferred_loc}",
                    "Master core financial modeling mechanics: 3-statement integrated models, dynamic Excel formulas, and scenario toggles",
                    "Develop intermediate SQL fluency: multi-table joins, aggregations, CTEs, and window functions on sample financial databases"
                ],
                action_items=[
                    "Dedicate 6 to 8 hours weekly to financial modeling exercises and practicing SQL queries on sample enterprise ledgers",
                    "Build a modular Excel model linking Income Statement, Balance Sheet, and Cash Flow with dynamic sensitivity analysis",
                    "Establish a GitHub or Notion portfolio workspace to document financial analyses, methodology notes, and query snippets"
                ]
            ),
            RoadmapPhase(
                month_range="Months 3-4",
                phase_title="Applied Portfolio Project: Financial Performance Dashboard & Valuation",
                focus_milestones=[
                    f"Design and publish an interactive Financial Performance & KPI Dashboard in Power BI, Tableau, or Python analyzing public corporate filings (10-K/10-Q) in {preferred_loc}",
                    "Build a Discounted Cash Flow (DCF) valuation and historical ratio analysis (liquidity, profitability, leverage) with transparent Bull/Base/Bear scenario cases",
                    "Author a 2-page executive summary memo interpreting variance drivers and operational recommendations as if presenting to an executive investment committee"
                ],
                action_items=[
                    "Devote 8 to 10 hours weekly to dashboard layout, DAX/Python metrics calculation, and financial data cleaning",
                    "Solicit constructive feedback on your financial assumptions and visual hierarchy from university professors, alumni, or finance mentors",
                    "Record a concise 3-minute video walkthrough explaining your model methodology, data sources, and business insights"
                ]
            ),
            RoadmapPhase(
                month_range="Months 5-6",
                phase_title="Executive Presentation, Finance Networking & Targeted Job Applications",
                focus_milestones=[
                    f"Incorporate portfolio dashboard metrics and valuation deliverables into a tailored resume for financial analyst roles in {preferred_loc}",
                    f"Conduct 3 to 5 informational interviews with finance, FP&A, or business intelligence professionals working in {preferred_loc}",
                    "Submit targeted applications for Financial Data Analyst, FP&A Analyst, and Business Intelligence Associate positions"
                ],
                action_items=[
                    "Attend regional campus finance career fairs, alumni networking panels, and local financial analyst forums",
                    "Engage in timed financial modeling tests and practice behavioral interview questions using the STAR framework",
                    f"Connect with finance practitioners on LinkedIn with personalized notes referencing shared academic interests in {major}"
                ]
            )
        ]

        executive_summary = (
            f"Local Career Guidance Synthesis: Based on your academic background in {major} ({study_level}) "
            f"and your aspiration to pursue {target_goal}, this deterministic rule-based evaluation highlights "
            f"three realistic career pathways tailored to your quantitative strengths. By prioritizing financial modeling, "
            f"SQL, and interactive dashboarding, you can build a competitive portfolio for opportunities in {preferred_loc} "
            "without requiring external AI services."
        )

    # -------------------------------------------------------------------------
    # DOMAIN BRANCH B: Other Disciplines (Tech, Life Sciences, Engineering, Design, General)
    # -------------------------------------------------------------------------
    else:
        has_git = has_user_skill(["git", "github", "version control"])
        has_python = has_user_skill(["python", "coding", "programming"])
        has_cloud = has_user_skill(["cloud", "aws", "gcp", "azure", "docker"])
        has_data = has_user_skill(["sql", "data analysis", "tableau", "statistics"])

        # Determine domain-specific gaps and resources
        if is_pure_tech_ai:
            candidate_gaps = []
            if not has_python:
                candidate_gaps.append((True, "Python Programming for Machine Learning (NumPy, pandas, scikit-learn)"))
            else:
                candidate_gaps.append((False, "Advanced Deep Learning Architectures (PyTorch, Transformers, Model Optimization)"))
            if not has_data:
                candidate_gaps.append((True, "Data Engineering Fundamentals: Relational databases, SQL, and feature stores"))
            else:
                candidate_gaps.append((False, "Production ML Systems: Model monitoring, drift detection, and MLflow pipeline tracking"))
            if not has_cloud:
                candidate_gaps.append((True, "Cloud AI Deployment: Serving models via REST APIs on Google Cloud or AWS"))
            else:
                candidate_gaps.append((False, "Scalable Distributed Training and GPU acceleration architectures"))

            candidate_gaps.sort(key=lambda x: not x[0])
            technical_gaps = [item[1] for item in candidate_gaps[:3]]

            resources = [
                LearningResource(
                    name="DeepLearning.AI Machine Learning Specialization",
                    resource_type="Online Course",
                    provider_or_platform="Coursera",
                    practical_value="Foundational curriculum covering supervised and unsupervised learning algorithms and practical ML engineering.",
                    url="https://www.coursera.org/specializations/machine-learning-introduction"
                ),
                LearningResource(
                    name="Google Cloud Digital Leader / Cloud Fundamentals",
                    resource_type="Certification",
                    provider_or_platform="Google Cloud",
                    practical_value="Validates foundational understanding of cloud infrastructure, data services, and security principles.",
                    url="https://cloud.google.com/learn/certification/cloud-digital-leader"
                ),
                LearningResource(
                    name="Harvard CS50: Introduction to Computer Science",
                    resource_type="Online Course",
                    provider_or_platform="edX",
                    practical_value="Rigorous computer science curriculum reinforcing data structures, algorithmic complexity, and systems programming.",
                    url="https://www.edx.org/cs50"
                )
            ]

        elif is_software_tech:
            candidate_gaps = []
            if not has_git:
                candidate_gaps.append((True, "Modern Version Control Workflows: Git branching strategies, pull requests, and code reviews"))
            else:
                candidate_gaps.append((False, "Advanced CI/CD Automation: GitHub Actions workflows, test automation, and release management"))
            if not has_cloud:
                candidate_gaps.append((True, "Containerization & Cloud Basics: Docker fundamentals and cloud container deployment"))
            else:
                candidate_gaps.append((False, "Microservice Architectures, Kubernetes orchestration, and observability"))
            candidate_gaps.append((True, "Modular Software Design, automated unit testing, and REST API development"))

            candidate_gaps.sort(key=lambda x: not x[0])
            technical_gaps = [item[1] for item in candidate_gaps[:3]]

            resources = [
                LearningResource(
                    name="Meta Front-End Developer Professional Certificate",
                    resource_type="Certification",
                    provider_or_platform="Coursera / Meta",
                    practical_value="Comprehensive curriculum covering JavaScript, React, version control, and responsive web applications.",
                    url="https://www.coursera.org/professional-certificates/meta-front-end-developer"
                ),
                LearningResource(
                    name="Google Cloud Digital Leader",
                    resource_type="Certification",
                    provider_or_platform="Google Cloud",
                    practical_value="Validates understanding of core cloud services, container workloads, and security fundamentals.",
                    url="https://cloud.google.com/learn/certification/cloud-digital-leader"
                ),
                LearningResource(
                    name="Harvard CS50: Introduction to Computer Science",
                    resource_type="Online Course",
                    provider_or_platform="edX",
                    practical_value="Strengthens core computational thinking, C/Python programming, memory management, and web engineering.",
                    url="https://www.edx.org/cs50"
                )
            ]

        elif is_health_bio:
            technical_gaps = [
                "Good Clinical Practice (GCP) and regulatory compliance frameworks (FDA / EMA guidelines)",
                "Biostatistical analysis and experimental data curation software (R / GraphPad)",
                "Electronic Data Capture (EDC) systems and Laboratory Information Management Systems (LIMS)"
            ]
            resources = [
                LearningResource(
                    name="Genomic Data Science Specialization",
                    resource_type="Online Course",
                    provider_or_platform="Coursera / Johns Hopkins University",
                    practical_value="Covers computational tools, Bioconductor in R, and statistical genomics for biomedical research.",
                    url="https://www.coursera.org/specializations/genomic-data-science"
                ),
                LearningResource(
                    name="Good Clinical Practice (GCP) Certification",
                    resource_type="Certification",
                    provider_or_platform="CITI Program",
                    practical_value="Demonstrates baseline compliance literacy required for clinical trial research and laboratory data coordination.",
                    url="https://about.citiprogram.org/course/good-clinical-practice-gcp/"
                ),
                LearningResource(
                    name="Google Data Analytics Professional Certificate",
                    resource_type="Certification",
                    provider_or_platform="Coursera / Google",
                    practical_value="Develops quantitative data cleaning, SQL, and data visualization capabilities for clinical datasets.",
                    url="https://www.coursera.org/professional-certificates/google-data-analytics"
                )
            ]

        elif is_engineering:
            technical_gaps = [
                "Industry-standard 3D CAD modeling and geometric dimensioning & tolerancing (GD&T)",
                "Finite Element Analysis (FEA) and multiphysics simulation software (ANSYS / SolidWorks Simulation)",
                "Engineering change orders, DFMEA documentation, and quality assurance standards (ISO / IEEE)"
            ]
            resources = [
                LearningResource(
                    name="Autodesk CAD/CAM for Manufacturing Specialization",
                    resource_type="Online Course",
                    provider_or_platform="Coursera / Autodesk",
                    practical_value="Develops applied 3D parametric CAD modeling, design for manufacturing (DFM), and simulation validation.",
                    url="https://www.coursera.org/specializations/autodesk-cad-cam-manufacturing"
                ),
                LearningResource(
                    name="Google Project Management Professional Certificate",
                    resource_type="Certification",
                    provider_or_platform="Coursera / Google",
                    practical_value="Builds core competencies in project documentation, risk mitigation, and technical execution methodologies.",
                    url="https://www.coursera.org/professional-certificates/google-project-management"
                ),
                LearningResource(
                    name="Harvard CS50: Introduction to Computer Science",
                    resource_type="Online Course",
                    provider_or_platform="edX",
                    practical_value="Strengthens computational literacy, scripting, and automation tools for engineering design workflows.",
                    url="https://www.edx.org/cs50"
                )
            ]

        else:
            # General Business / Humanities / Social Sciences
            technical_gaps = [
                "Quantitative data analysis and spreadsheet modeling (Advanced Excel / SQL fundamentals)",
                "Agile project management and issue-tracking platforms (Jira / Asana / Confluence)",
                "Digital artifact documentation, presentation design, and workflow automation"
            ]
            resources = [
                LearningResource(
                    name="Google Project Management Professional Certificate",
                    resource_type="Certification",
                    provider_or_platform="Coursera / Google",
                    practical_value="Builds core competencies in project documentation, risk mitigation, and Agile/Scrum execution methodologies.",
                    url="https://www.coursera.org/professional-certificates/google-project-management"
                ),
                LearningResource(
                    name="Google Data Analytics Professional Certificate",
                    resource_type="Certification",
                    provider_or_platform="Coursera / Google",
                    practical_value="Provides end-to-end training in data cleaning, SQL database querying, Tableau, and data-driven storytelling.",
                    url="https://www.coursera.org/professional-certificates/google-data-analytics"
                ),
                LearningResource(
                    name="Wharton Business Analytics Specialization",
                    resource_type="Online Course",
                    provider_or_platform="Coursera / Wharton School",
                    practical_value="Develops quantitative fluency, data-driven reasoning, and strategic decision-making capabilities.",
                    url="https://www.coursera.org/specializations/business-analytics"
                )
            ]

        soft_skills = [
            "Technical communication: translating complex domain concepts for non-specialist stakeholders",
            "Structured task prioritization: managing competing deadlines across multi-month projects",
            f"Professional networking: conducting informational interviews and engaging communities in {preferred_loc}"
        ]

        rec1_title = target_goal if any(w in target_goal.lower() for w in ["engineer", "analyst", "developer", "specialist", "scientist", "manager", "consultant", "researcher", "coordinator"]) else f"Associate {target_goal}"
        rec1 = CareerRecommendation(
            role_title=rec1_title,
            rationale=(
                f"Directly advances your primary career goal as a {target_goal}. "
                f"Your academic foundation in {major} ({study_level}) provides theoretical rigor, "
                f"and your interest in {profile.interests} offers domain context for opportunities in {preferred_loc}."
            ),
            typical_responsibilities=[
                f"Contribute to day-to-day deliverables and implementation tasks related to {target_goal}",
                "Collaborate with senior practitioners and cross-functional teams to resolve technical and operational challenges",
                "Document workflows, test deliverables, and conduct rigorous peer reviews",
                f"Align project milestones with organizational requirements in {preferred_loc}"
            ],
            transferable_skills=get_transferable(["Analytical Thinking", "Structured Problem Solving", "Academic Research"], 3),
            growth_areas=[
                f"Mastering industry-standard tools and frameworks specific to {target_goal}",
                "Developing end-to-end portfolio projects with production-grade documentation",
                "Navigating professional quality assurance and team collaboration workflows"
            ]
        )

        rec2 = CareerRecommendation(
            role_title="Data & Operations Solutions Analyst",
            rationale=(
                f"An adjacent, high-demand analytical path that pairs your {major} background "
                f"with data processing and systems integration in {preferred_loc}."
            ),
            typical_responsibilities=[
                "Query, clean, and model datasets to support operational decision-making",
                "Build automated data verification pipelines and maintain dashboard telemetry",
                "Partner with engineering teams to ensure data integrity and workflow reliability"
            ],
            transferable_skills=get_transferable(["Data Analysis", "Communication"], 2),
            growth_areas=[
                "Database querying and schema design (SQL)",
                "Data visualization (Tableau / Power BI)",
                "Workflow automation and process optimization"
            ]
        )

        rec3 = CareerRecommendation(
            role_title="Technical Project & Implementation Coordinator",
            rationale=(
                f"A versatile cross-functional role pairing your {major} training with client-facing solution delivery in {preferred_loc}."
            ),
            typical_responsibilities=[
                "Track project deliverables, milestones, and cross-functional schedules",
                "Translate client requirements into actionable project deliverables",
                "Maintain project documentation and conduct status reviews with stakeholders"
            ],
            transferable_skills=get_transferable(["Project Coordination", "Stakeholder Management"], 2),
            growth_areas=[
                "Agile / Scrum methodologies (Jira)",
                "Resource tracking and risk mitigation planning",
                "Executive stakeholder presentation"
            ]
        )

        roadmap = [
            RoadmapPhase(
                month_range="Months 1-2",
                phase_title="Foundations, Skill Benchmarking & Environment Setup",
                focus_milestones=[
                    f"Audit academic coursework in {major} against entry-level job requisitions in {preferred_loc}",
                    "Establish a version-controlled portfolio environment (GitHub profile, workspace tooling, and study tracker)",
                    "Complete the first module of a targeted foundational certification or specialization"
                ],
                action_items=[
                    "Dedicate 5 to 8 hours weekly to focused skill-building and technical exercises",
                    "Publish all laboratory exercises or mini-project deliverables to your public portfolio repository",
                    "Join at least one professional student organization or online discipline-specific community"
                ]
            ),
            RoadmapPhase(
                month_range="Months 3-4",
                phase_title="Applied Portfolio Project & Practical Implementation",
                focus_milestones=[
                    f"Design and execute a comprehensive capstone project addressing a realistic challenge in {target_goal}",
                    "Write rigorous documentation, user guides, and unit tests validating your project deliverables",
                    "Solicit constructive feedback on your project from university faculty, alumni, or online mentors"
                ],
                action_items=[
                    "Devote 8 to 10 hours weekly to iterative project development and deliverable refactoring",
                    "Record a concise 3-minute video walkthrough summarizing your methodology and findings",
                    "Publish an explanatory article or technical README detailing the problem solved and lessons learned"
                ]
            ),
            RoadmapPhase(
                month_range="Months 5-6",
                phase_title="Professional Networking, Resume Refinement & Career Launch",
                focus_milestones=[
                    f"Incorporate project achievements and quantified outcomes into a targeted resume for roles in {preferred_loc}",
                    f"Conduct 3 to 5 informational interviews with professionals working in {target_goal}",
                    "Submit targeted applications for internships, graduate development programs, or associate roles"
                ],
                action_items=[
                    "Attend regional campus career fairs, industry meetups, or virtual employer showcases",
                    "Engage with alumni on LinkedIn with personalized notes referencing shared academic interests",
                    "Practice technical and behavioral interview scenarios using the STAR method"
                ]
            )
        ]

        executive_summary = (
            f"Local Career Guidance Synthesis: Based on your academic background in {major} ({study_level}) "
            f"and your aspiration to pursue {target_goal}, this deterministic rule-based evaluation highlights "
            f"three realistic career pathways. By building hands-on portfolio projects and bridging key technical "
            f"competencies, you can prepare an actionable trajectory for opportunities in {preferred_loc} "
            "without requiring external AI services."
        )

    # -------------------------------------------------------------------------
    # Shared Strengths & Advisory Assembly
    # -------------------------------------------------------------------------
    identified_strengths = [
        f"Academic training in {major} at the {study_level} level"
    ]
    if skills_items:
        identified_strengths.append(f"Demonstrated competencies: {', '.join(skills_items[:4])}")
    else:
        identified_strengths.append("Foundational analytical inquiry and structured university coursework")
    identified_strengths.append(f"Clear career orientation toward roles as a {target_goal}")

    skills_gap = SkillsGapAnalysis(
        identified_strengths=identified_strengths,
        technical_skills_to_acquire=technical_gaps,
        soft_skills_recommendations=soft_skills
    )

    advisory_note = (
        "Local Career Guidance Notice: This assessment is generated via transparent rule-based matching based "
        "on current academic and industry frameworks. Hiring outcomes depend on portfolio depth, interview performance, "
        "and regional market demand. Certifications develop verifiable competencies but do not guarantee employment."
    )

    return CareerAnalysisReport(
        executive_summary=executive_summary,
        career_recommendations=[rec1, rec2, rec3],
        skills_gap=skills_gap,
        learning_resources=resources,
        six_month_roadmap=roadmap,
        advisory_note=advisory_note
    )
