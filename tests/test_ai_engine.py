"""
Automated Unit Tests - AI Engine
--------------------------------
These tests verify that user profile validation, prompt generation,
input checks, Google Gemini API integration (via mocked google-genai client),
error handling, and the offline Local Career Guidance fallback engine work
correctly WITHOUT making any live API calls or requiring an internet connection.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root directory to sys.path so tests can import ai_engine directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic import ValidationError
from google.genai.errors import ClientError, ServerError
import httpx

from ai_engine import (
    UserProfile,
    build_recommendation_prompt,
    validate_api_key,
    parse_comma_separated_list,
    validate_student_input,
    generate_career_guidance,
    generate_local_guidance,
    CareerRecommendation,
    SkillsGapAnalysis,
    LearningResource,
    RoadmapPhase,
    CareerAnalysisReport,
    DEFAULT_GEMINI_MODEL,
    MissingAPIKeyError,
    AuthenticationAPIError,
    RateLimitAPIError,
    NetworkAPIError,
    InvalidResponseError,
    CareerGuidanceError,
)


# -----------------------------------------------------------------------------
# Existing Baseline Validation Tests (Preserved)
# -----------------------------------------------------------------------------
def test_user_profile_creation_valid():
    """Verify that a student profile can be successfully created with valid inputs."""
    profile = UserProfile(
        academic_background="Computer Science, 3rd Year",
        skills="Python, SQL, Problem Solving",
        interests="Machine Learning, Data Science",
        goals="Become a Machine Learning Engineer",
        preferred_location="New York, NY or Remote"
    )
    assert profile.academic_background == "Computer Science, 3rd Year"
    assert profile.study_level == "Undergraduate"  # Default value
    assert profile.skills == "Python, SQL, Problem Solving"
    assert profile.interests == "Machine Learning, Data Science"
    assert profile.goals == "Become a Machine Learning Engineer"
    assert profile.preferred_location == "New York, NY or Remote"


def test_user_profile_with_custom_study_level():
    """Verify that custom study level is properly stored."""
    profile = UserProfile(
        academic_background="Data Science",
        study_level="Master's",
        skills="Python, R, Statistics",
        interests="Big Data, Cloud Computing",
        goals="Data Architect",
        preferred_location="Canada"
    )
    assert profile.study_level == "Master's"


def test_user_profile_missing_field():
    """Verify that omitting a required field raises a validation error."""
    with pytest.raises(ValidationError):
        # Missing 'preferred_location'
        UserProfile(
            academic_background="Biology",
            skills="Lab techniques",
            interests="Genetics",
            goals="Biotech Researcher"
        )


def test_build_recommendation_prompt():
    """Verify that prompt builder properly includes user profile details."""
    profile = UserProfile(
        academic_background="Economics",
        study_level="Undergraduate",
        skills="Data Analysis, Excel, R",
        interests="Financial Markets, Fintech",
        goals="Quantitative Analyst",
        preferred_location="London or Remote"
    )
    prompt = build_recommendation_prompt(profile)

    assert "Economics" in prompt
    assert "Undergraduate" in prompt
    assert "Financial Markets, Fintech" in prompt
    assert "Quantitative Analyst" in prompt
    assert "London or Remote" in prompt


def test_validate_api_key():
    """Verify Gemini API key validation logic including placeholder checks."""
    assert validate_api_key(None) is False
    assert validate_api_key("") is False
    assert validate_api_key("   ") is False
    assert validate_api_key("your-gemini-api-key-here") is False
    assert validate_api_key("your-openai-api-key-here") is False
    assert validate_api_key("AIzaSyTestKey123456789") is True


def test_parse_comma_separated_list():
    """Verify string parsing, trimming, and case-insensitive deduplication."""
    assert parse_comma_separated_list("") == []
    assert parse_comma_separated_list("   ") == []

    result = parse_comma_separated_list("Python,  SQL, python, Machine Learning, , Data Analysis")
    assert result == ["Python", "SQL", "Machine Learning", "Data Analysis"]


def test_validate_student_input():
    """Verify required field validation for major and career goal."""
    is_valid, error = validate_student_input("Computer Science", "Software Engineer")
    assert is_valid is True
    assert error is None

    is_valid, error = validate_student_input("", "Software Engineer")
    assert is_valid is False
    assert "major" in error.lower()

    is_valid, error = validate_student_input("A", "Software Engineer")
    assert is_valid is False
    assert "major" in error.lower()

    is_valid, error = validate_student_input("Physics", "")
    assert is_valid is False
    assert "career goal" in error.lower()


# -----------------------------------------------------------------------------
# Gemini Model Constants & Schema Constraint Tests
# -----------------------------------------------------------------------------
def test_default_model_constant():
    """Verify that the default Gemini model alias is set to gemini-2.5-flash."""
    assert DEFAULT_GEMINI_MODEL == "gemini-2.5-flash"


def _build_sample_mock_report() -> CareerAnalysisReport:
    """Helper creating a valid mock report adhering to the required 3 paths and 3 roadmap phases."""
    return CareerAnalysisReport(
        executive_summary="Solid foundation in computer science with strong potential in backend engineering.",
        career_recommendations=[
            CareerRecommendation(
                role_title="Junior Machine Learning Engineer",
                rationale="Aligns with Python proficiency and machine learning coursework.",
                typical_responsibilities=["Train ML models", "Deploy APIs", "Monitor data pipelines"],
                transferable_skills=["Python", "SQL"],
                growth_areas=["Model monitoring", "Containerization (Docker)"]
            ),
            CareerRecommendation(
                role_title="Data Engineer",
                rationale="Builds on SQL and data processing background.",
                typical_responsibilities=["Maintain ETL pipelines", "Data warehousing", "Query optimization"],
                transferable_skills=["SQL", "Database design"],
                growth_areas=["Apache Spark", "Airflow"]
            ),
            CareerRecommendation(
                role_title="Applied AI Solutions Developer",
                rationale="Leverages application development skills to integrate LLM workflows.",
                typical_responsibilities=["Integrate AI APIs", "Build user interfaces", "Evaluate prompts"],
                transferable_skills=["Python", "Web APIs"],
                growth_areas=["LangChain", "Vector databases"]
            )
        ],
        skills_gap=SkillsGapAnalysis(
            identified_strengths=["Strong programming logic", "Database querying"],
            technical_skills_to_acquire=["Docker", "Cloud computing (AWS/GCP)"],
            soft_skills_recommendations=["Technical communication", "Cross-functional teamwork"]
        ),
        learning_resources=[
            LearningResource(
                name="Google Cloud Digital Leader",
                resource_type="Certification",
                provider_or_platform="Google Cloud",
                practical_value="Demonstrates baseline cloud literacy (note: builds competency, not a job guarantee)."
            ),
            LearningResource(
                name="DeepLearning.AI Machine Learning Specialization",
                resource_type="Online Course",
                provider_or_platform="Coursera",
                practical_value="Deepens supervised and unsupervised learning algorithms."
            )
        ],
        six_month_roadmap=[
            RoadmapPhase(
                month_range="Months 1-2",
                phase_title="Core Cloud & Containerization Foundations",
                focus_milestones=["Complete Docker basics", "Build a containerized Python API"],
                action_items=["Dedicate 5 hours/week to hands-on labs", "Push projects to GitHub"]
            ),
            RoadmapPhase(
                month_range="Months 3-4",
                phase_title="Applied Portfolio Project Development",
                focus_milestones=["Build and deploy an end-to-end ML service", "Write comprehensive documentation"],
                action_items=["Write unit tests", "Record a 2-minute demo video"]
            ),
            RoadmapPhase(
                month_range="Months 5-6",
                phase_title="Mentorship, Networking & Career Preparation",
                focus_milestones=["Conduct informational interviews", "Refine resume with campus advisors"],
                action_items=["Attend local tech meetups", "Apply to targeted graduate roles"]
            )
        ],
        advisory_note="Career development requires continuous practice. Certifications build skills but do not guarantee hiring outcomes."
    )


def test_career_analysis_report_schema_enforces_three_recommendations():
    """Verify that fewer or more than 3 recommendations violates the schema constraints."""
    sample = _build_sample_mock_report()

    # Valid schema works
    assert len(sample.career_recommendations) == 3
    assert len(sample.six_month_roadmap) == 3

    # Attempting only 2 recommendations must raise ValidationError
    with pytest.raises(ValidationError):
        CareerAnalysisReport(
            executive_summary=sample.executive_summary,
            career_recommendations=sample.career_recommendations[:2],  # only 2
            skills_gap=sample.skills_gap,
            learning_resources=sample.learning_resources,
            six_month_roadmap=sample.six_month_roadmap,
            advisory_note=sample.advisory_note
        )


# -----------------------------------------------------------------------------
# Mocked Gemini API Integration Tests (Zero Paid / Live Calls)
# -----------------------------------------------------------------------------
def test_generate_career_guidance_missing_key(monkeypatch):
    """Verify that attempting generation without a Gemini API key raises MissingAPIKeyError immediately."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    profile = UserProfile(
        academic_background="Computer Science",
        skills="Python, SQL",
        interests="AI",
        goals="Developer",
        preferred_location="Remote"
    )

    with pytest.raises(MissingAPIKeyError):
        generate_career_guidance(profile=profile, api_key=None)

    with pytest.raises(MissingAPIKeyError):
        generate_career_guidance(profile=profile, api_key="")

    with pytest.raises(MissingAPIKeyError):
        generate_career_guidance(profile=profile, api_key="your-gemini-api-key-here")


@patch("google.genai.Client")
def test_generate_career_guidance_success(mock_client_class):
    """Verify that generate_career_guidance parses the mock Gemini response and returns report."""
    expected_report = _build_sample_mock_report()

    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.parsed = expected_report
    mock_response.text = expected_report.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    profile = UserProfile(
        academic_background="Computer Science",
        study_level="Undergraduate",
        skills="Python, SQL",
        interests="Machine Learning",
        goals="ML Engineer",
        preferred_location="New York"
    )

    result = generate_career_guidance(profile=profile, api_key="AIzaSyMockValidKey")

    assert isinstance(result, CareerAnalysisReport)
    assert len(result.career_recommendations) == 3
    assert result.career_recommendations[0].role_title == "Junior Machine Learning Engineer"
    assert len(result.six_month_roadmap) == 3
    assert "Months 1-2" in result.six_month_roadmap[0].month_range

    # Verify generate_content was called with correct model and structured output schema
    mock_client.models.generate_content.assert_called_once()
    kwargs = mock_client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "gemini-2.5-flash"
    assert kwargs["config"].response_schema == CareerAnalysisReport
    assert kwargs["config"].response_mime_type == "application/json"


@patch("google.genai.Client")
def test_generate_career_guidance_json_text_fallback(mock_client_class):
    """Verify that if response.parsed is None, the engine parses response.text JSON string."""
    expected_report = _build_sample_mock_report()

    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.parsed = None
    mock_response.output_parsed = None
    mock_response.text = expected_report.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    profile = UserProfile(
        academic_background="Data Science",
        study_level="Master's",
        skills="Python, R",
        interests="Big Data",
        goals="Data Architect",
        preferred_location="Remote"
    )

    result = generate_career_guidance(profile=profile, api_key="AIzaSyMockValidKey")
    assert isinstance(result, CareerAnalysisReport)
    assert len(result.career_recommendations) == 3


@patch("google.genai.Client")
def test_generate_career_guidance_authentication_error(mock_client_class):
    """Verify that a 403 / PERMISSION_DENIED ClientError raises AuthenticationAPIError."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_client.models.generate_content.side_effect = ClientError(
        403,
        {"error": {"code": 403, "message": "API key not valid. Please pass a valid API key.", "status": "PERMISSION_DENIED"}}
    )

    profile = UserProfile(
        academic_background="Mathematics",
        skills="Calculus, Python",
        interests="Statistics",
        goals="Actuary",
        preferred_location="Remote"
    )

    with pytest.raises(AuthenticationAPIError) as exc_info:
        generate_career_guidance(profile=profile, api_key="AIzaSyMockInvalidKey")

    assert "invalid or unauthorized" in str(exc_info.value).lower()


@patch("google.genai.Client")
def test_generate_career_guidance_rate_limit_error(mock_client_class):
    """Verify that a 429 / RESOURCE_EXHAUSTED ClientError raises RateLimitAPIError."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_client.models.generate_content.side_effect = ClientError(
        429,
        {"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota).", "status": "RESOURCE_EXHAUSTED"}}
    )

    profile = UserProfile(
        academic_background="Business",
        skills="Excel, Accounting",
        interests="Consulting",
        goals="Consultant",
        preferred_location="Chicago"
    )

    with pytest.raises(RateLimitAPIError) as exc_info:
        generate_career_guidance(profile=profile, api_key="AIzaSyMockKey")

    assert "rate limit or quota" in str(exc_info.value).lower()


@patch("google.genai.Client")
def test_generate_career_guidance_network_error(mock_client_class):
    """Verify that 5xx ServerError or network timeouts raise NetworkAPIError."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_client.models.generate_content.side_effect = ServerError(
        503,
        {"error": {"code": 503, "message": "The service is temporarily unavailable.", "status": "UNAVAILABLE"}}
    )

    profile = UserProfile(
        academic_background="Engineering",
        skills="CAD, MATLAB",
        interests="Robotics",
        goals="Robotics Engineer",
        preferred_location="Boston"
    )

    with pytest.raises(NetworkAPIError) as exc_info:
        generate_career_guidance(profile=profile, api_key="AIzaSyMockKey")

    assert "could not connect" in str(exc_info.value).lower()


@patch("google.genai.Client")
def test_generate_career_guidance_invalid_response_schema(mock_client_class):
    """Verify that empty or malformed model responses raise InvalidResponseError."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.parsed = None
    mock_response.output_parsed = None
    mock_response.text = "{}"
    mock_client.models.generate_content.return_value = mock_response

    profile = UserProfile(
        academic_background="Physics",
        skills="Optics",
        interests="Quantum",
        goals="Research Scientist",
        preferred_location="Remote"
    )

    with pytest.raises(InvalidResponseError):
        generate_career_guidance(profile=profile, api_key="AIzaSyMockKey")


# -----------------------------------------------------------------------------
# Local Fallback Career Guidance Engine Tests (No API Key Required)
# -----------------------------------------------------------------------------
def test_local_fallback_no_api_key_required(monkeypatch):
    """Verify that local fallback executes completely offline without any API key or env var."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    profile = UserProfile(
        academic_background="Computer Science",
        study_level="Undergraduate",
        skills="Python, SQL, Algorithms",
        interests="Cloud Computing, AI",
        goals="Full Stack Developer",
        preferred_location="Remote"
    )

    report = generate_local_guidance(profile=profile)
    assert report is not None
    assert isinstance(report, CareerAnalysisReport)


def test_local_fallback_contains_exactly_three_recommendations():
    """Verify that the local fallback engine strictly returns exactly 3 recommendations."""
    profile = UserProfile(
        academic_background="Data Analytics",
        study_level="Master's",
        skills="R, Tableau, Statistics",
        interests="Business Intelligence",
        goals="Data Analyst",
        preferred_location="New York"
    )

    report = generate_local_guidance(profile=profile)
    assert len(report.career_recommendations) == 3

    for rec in report.career_recommendations:
        assert isinstance(rec, CareerRecommendation)
        assert len(rec.role_title.strip()) > 0
        assert len(rec.rationale.strip()) > 0
        assert len(rec.typical_responsibilities) >= 2
        assert len(rec.transferable_skills) >= 1
        assert len(rec.growth_areas) >= 2


def test_local_fallback_preserves_schema():
    """Verify that the local fallback output satisfies all CareerAnalysisReport schema requirements."""
    profile = UserProfile(
        academic_background="Biomedical Science",
        study_level="Undergraduate",
        skills="Lab Research, PCR, Data Analysis",
        interests="Genetics, Pharmacology",
        goals="Clinical Research Coordinator",
        preferred_location="Boston, MA"
    )

    report = generate_local_guidance(profile=profile)

    assert isinstance(report.skills_gap, SkillsGapAnalysis)
    assert len(report.skills_gap.identified_strengths) >= 2
    assert len(report.skills_gap.technical_skills_to_acquire) >= 2
    assert len(report.skills_gap.soft_skills_recommendations) >= 2

    assert len(report.learning_resources) >= 2
    for res in report.learning_resources:
        assert isinstance(res, LearningResource)
        assert len(res.name) > 0
        assert len(res.provider_or_platform) > 0

    assert len(report.six_month_roadmap) == 3
    expected_phases = ["Months 1-2", "Months 3-4", "Months 5-6"]
    for idx, phase in enumerate(report.six_month_roadmap):
        assert isinstance(phase, RoadmapPhase)
        assert phase.month_range == expected_phases[idx]
        assert len(phase.focus_milestones) >= 2
        assert len(phase.action_items) >= 2


def test_local_fallback_clearly_labeled_local_guidance():
    """Verify that the local fallback is clearly labeled as Local Career Guidance."""
    profile = UserProfile(
        academic_background="Finance",
        skills="Excel, Accounting",
        interests="Venture Capital",
        goals="Investment Analyst",
        preferred_location="London"
    )

    report = generate_local_guidance(profile=profile)

    assert "local career guidance" in report.executive_summary.lower()
    assert "local career guidance" in report.advisory_note.lower()


def test_local_fallback_no_invented_statistics_or_salaries():
    """Verify that local guidance does not invent fake salaries, guarantees, or precision match percentages."""
    profile = UserProfile(
        academic_background="Mechanical Engineering",
        skills="SolidWorks, MATLAB, Thermodynamics",
        interests="Automotive, Clean Energy",
        goals="Mechanical Design Engineer",
        preferred_location="Germany"
    )

    report = generate_local_guidance(profile=profile)

    full_text = (
        report.executive_summary
        + " "
        + report.advisory_note
        + " "
        + " ".join(r.rationale for r in report.career_recommendations)
    ).lower()

    # Must never invent salary numbers or fake match percentages
    assert "$ " not in full_text
    assert "% match" not in full_text
    assert "guarantee" not in report.executive_summary.lower()


def test_local_fallback_diverse_majors():
    """Verify that the rule-based local engine dynamically adapts to diverse fields of study."""
    diverse_test_cases = [
        ("Computer Science", "Software Engineer"),
        ("Biology", "Biotech Researcher"),
        ("Finance", "Financial Analyst"),
        ("Psychology", "HR Specialist"),
        ("Graphic Design", "UI/UX Designer"),
        ("History", "Policy Advisor"),
    ]

    for major, goal in diverse_test_cases:
        profile = UserProfile(
            academic_background=major,
            skills="Communication, Critical Thinking",
            interests="Industry Innovations",
            goals=goal,
            preferred_location="Remote"
        )
        report = generate_local_guidance(profile=profile)
        assert len(report.career_recommendations) == 3
        assert len(report.six_month_roadmap) == 3
        assert report.career_recommendations[0].role_title


# -----------------------------------------------------------------------------
# Financial Data Analysis & Domain-Specific Personalization Tests
# -----------------------------------------------------------------------------
def test_local_fallback_financial_data_analyst_prioritizes_finance_skills():
    """Verify that a Finance / Management Science student aiming for Financial Data Analyst
    receives finance-appropriate skill gaps and NOT software engineering DevOps tools."""
    profile = UserProfile(
        academic_background="Management Science and Finance",
        study_level="Undergraduate",
        skills="Excel, Financial Analysis, Accounting",
        interests="Corporate Finance, Investment",
        goals="Financial Data Analyst",
        preferred_location="New York, NY"
    )

    report = generate_local_guidance(profile=profile)
    gaps_text = " ".join(report.skills_gap.technical_skills_to_acquire).lower()

    # Must prioritize financial data analysis skills
    assert any(k in gaps_text for k in ["financial modeling", "sql", "power bi", "tableau", "python", "statistics"])

    # Must NOT recommend software development DevOps infrastructure tools as default gaps
    assert "docker" not in gaps_text
    assert "ci/cd" not in gaps_text
    assert "github actions" not in gaps_text
    assert "kubernetes" not in gaps_text
    assert "containerization" not in gaps_text


def test_local_fallback_does_not_recommend_existing_skills_as_basic_gaps():
    """Verify that existing skills are not recommended as basic gaps, and next levels are suggested when appropriate."""
    # Case 1: User already lists Excel -> Excel is not recommended as an unpossessed basic gap
    profile_has_excel = UserProfile(
        academic_background="Finance",
        study_level="Undergraduate",
        skills="Excel, Financial Analysis",
        interests="Data Analysis",
        goals="Financial Data Analyst",
        preferred_location="London"
    )
    report1 = generate_local_guidance(profile=profile_has_excel)
    gaps1 = report1.skills_gap.technical_skills_to_acquire

    # The prioritized gaps should be unpossessed skills like Financial Modeling, SQL, Power BI
    gaps1_lower = " ".join(gaps1).lower()
    assert "financial modeling" in gaps1_lower
    assert "sql" in gaps1_lower

    # Case 2: User already lists all core skills -> recommendations suggest logical next level
    profile_advanced = UserProfile(
        academic_background="Management Science and Finance",
        study_level="Master's",
        skills="Excel, Financial Modeling, SQL, Power BI, Python",
        interests="Quantitative Finance",
        goals="Financial Data Analyst",
        preferred_location="New York"
    )
    report2 = generate_local_guidance(profile=profile_advanced)
    gaps2 = report2.skills_gap.technical_skills_to_acquire

    # For skills already held, any suggestion must be framed as Next Level / Advanced progression
    for gap in gaps2:
        gap_lower = gap.lower()
        if "financial modeling" in gap_lower:
            assert "next level" in gap_lower or "advanced" in gap_lower
        if "excel" in gap_lower:
            assert "next level" in gap_lower or "advanced" in gap_lower
        if "sql" in gap_lower:
            assert "next level" in gap_lower or "advanced" in gap_lower


def test_local_fallback_finance_roadmap_includes_financial_project():
    """Verify that the six-month roadmap includes an applied financial dashboard or valuation project."""
    profile = UserProfile(
        academic_background="Management Science and Finance",
        study_level="Undergraduate",
        skills="Excel, Accounting",
        interests="Fintech, Investment",
        goals="Financial Data Analyst",
        preferred_location="Chicago, IL"
    )

    report = generate_local_guidance(profile=profile)
    roadmap_text = " ".join(
        phase.phase_title + " " + " ".join(phase.focus_milestones) + " " + " ".join(phase.action_items)
        for phase in report.six_month_roadmap
    ).lower()

    # Must feature a practical financial dashboard or financial data analysis project
    assert any(k in roadmap_text for k in ["financial performance", "dashboard", "valuation", "dcf", "financial modeling"])


def test_local_fallback_learning_resources_have_verifiable_urls():
    """Verify that learning resources provide genuine, verifiable URLs."""
    profile = UserProfile(
        academic_background="Management Science and Finance",
        study_level="Undergraduate",
        skills="Excel, Financial Analysis",
        interests="Corporate Finance",
        goals="Financial Data Analyst",
        preferred_location="Remote"
    )

    report = generate_local_guidance(profile=profile)
    assert len(report.learning_resources) >= 3

    for res in report.learning_resources:
        assert res.url is not None
        assert res.url.startswith("https://")
        assert any(domain in res.url for domain in ["coursera.org", "corporatefinanceinstitute.com", "edx.org", "microsoft.com"])


def test_local_fallback_finance_career_recommendations_tailored():
    """Verify that the 3 career recommendations are directly tailored to finance/business analytics."""
    profile = UserProfile(
        academic_background="Management Science and Finance",
        study_level="Undergraduate",
        skills="Excel, Financial Analysis, Statistics",
        interests="Corporate Finance, Investment",
        goals="Financial Data Analyst",
        preferred_location="New York, NY"
    )

    report = generate_local_guidance(profile=profile)
    assert len(report.career_recommendations) == 3

    titles = [r.role_title for r in report.career_recommendations]
    assert titles == [
        "Financial Data Analyst",
        "Business Intelligence Analyst — Finance & Strategy",
        "Corporate Financial Planning & Analysis (FP&A) Analyst"
    ]

    for rec in report.career_recommendations:
        assert len(rec.typical_responsibilities) >= 3
        assert len(rec.transferable_skills) >= 1
        assert len(rec.growth_areas) >= 2
        # Rationale must mention background or goal
        assert "Management Science and Finance" in rec.rationale or "finance" in rec.rationale.lower()


def test_exact_expected_three_career_titles_for_finance_profile():
    """Assert the exact expected three career titles for a Management Science and Finance
    undergraduate targeting Financial Data Analyst."""
    profile = UserProfile(
        academic_background="Management Science and Finance",
        study_level="Undergraduate",
        skills="Excel, Financial Analysis",
        interests="Corporate Finance",
        goals="Financial Data Analyst",
        preferred_location="New York, NY"
    )

    report = generate_local_guidance(profile=profile)
    titles = [r.role_title for r in report.career_recommendations]

    assert titles == [
        "Financial Data Analyst",
        "Business Intelligence Analyst — Finance & Strategy",
        "Corporate Financial Planning & Analysis (FP&A) Analyst"
    ]

