"""
AI Study & Career Navigator - Streamlit Application
---------------------------------------------------
A web application designed to help university students explore personalized
career paths based on their academic background, skills, interests, goals,
and preferred location.

Powered by Google's Gemini API (via the official google-genai SDK) with
structured Pydantic outputs, and an offline Local Career Guidance fallback
engine that works without any API key.
"""

import sys
from pathlib import Path
import importlib
import streamlit as st

# Ensure local project root is prioritized
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Force reload of ai_engine to prevent stale in-memory module caching across runs
import ai_engine
importlib.reload(ai_engine)

from ai_engine import (
    UserProfile,
    validate_student_input,
    parse_comma_separated_list,
    generate_career_guidance,
    generate_local_guidance,
    DEFAULT_GEMINI_MODEL,
    CareerAnalysisReport,
    CareerGuidanceError,
    MissingAPIKeyError,
    AuthenticationAPIError,
    RateLimitAPIError,
    NetworkAPIError,
    InvalidResponseError,
    validate_api_key,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Study & Career Navigator",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Sidebar Configuration & Secrets Resolution
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("About The Navigator")
    st.markdown(
        "**AI Study & Career Navigator** provides evidence-based academic "
        "pathways and career guidance using Google's official **Gemini API** "
        "(`google-genai`) with structured outputs, alongside an offline "
        "**Local Career Guidance** engine."
    )
    st.divider()

    st.subheader("⚙️ AI Configuration")
    st.caption(f"**Target Model:** `{DEFAULT_GEMINI_MODEL}` (Free Tier Supported)")

    # Resolve API Key securely
    # Priority 1: .streamlit/secrets.toml (GEMINI_API_KEY)
    # Priority 2: Optional sidebar input (for testing without saving to disk)
    configured_secret_key = ""
    try:
        configured_secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        configured_secret_key = ""

    sidebar_api_key = st.text_input(
        label="Gemini API Key (Session Override)",
        type="password",
        placeholder="AIzaSy...",
        help="Optional: If not set in .streamlit/secrets.toml, paste your Gemini API key here for this session."
    )

    # Active API key resolution (never printed or exposed)
    active_api_key = sidebar_api_key.strip() if sidebar_api_key.strip() else configured_secret_key
    has_gemini_key = validate_api_key(active_api_key)

    if has_gemini_key:
        st.success("🔑 Gemini API Key Detected", icon="✅")
    else:
        st.info("ℹ️ No Gemini API key detected. Local Career Guidance will be used.", icon="ℹ️")

    st.divider()
    st.caption("Engine: Google Gemini API & Local Rule Matching")
    st.caption("Version: 0.4.0")

# -----------------------------------------------------------------------------
# Main Header & Introduction
# -----------------------------------------------------------------------------
st.title("🎓 AI Study & Career Navigator")
st.markdown("""
Welcome! Planning your university education and career trajectory can be daunting.
This tool analyzes your coursework, skills, and personal ambitions to generate
**three personalized career pathways**, an in-depth **skills gap analysis**, curated **learning resources**,
and a structured **six-month action roadmap**.
""")

st.write("---")

# -----------------------------------------------------------------------------
# Interactive Student Profile Form
# -----------------------------------------------------------------------------
st.subheader("📝 Student Profile Questionnaire")
st.caption("Fields marked with an asterisk ( **\\* ** ) are required.")

with st.form(key="student_profile_form", clear_on_submit=False):
    # Section 1: Academic Background
    st.markdown("#### 1. Academic Background")
    col1, col2 = st.columns([3, 2])

    with col1:
        major = st.text_input(
            label="Academic Major / Field of Study *",
            placeholder="e.g., Computer Science, Biology, Economics, Psychology",
            help="Enter your primary major, program, or field of university study."
        )

    with col2:
        study_level = st.selectbox(
            label="Current Study Level",
            options=["Undergraduate", "Master's", "Doctoral", "Other"],
            index=0,
            help="Select your current degree or program level."
        )

    # Section 2: Skills & Interests
    st.markdown("#### 2. Skills & Interests")
    skills_raw = st.text_area(
        label="Skills (Comma-separated)",
        placeholder="e.g., Python, SQL, Financial Modeling, Public Speaking, Critical Thinking",
        help="List your technical, practical, or soft skills separated by commas."
    )

    interests_raw = st.text_area(
        label="Interests & Passions (Comma-separated)",
        placeholder="e.g., Artificial Intelligence, Climate Science, Quantitative Finance, Renewable Energy",
        help="List subjects, industries, or technologies that excite you."
    )

    # Section 3: Goals & Preferred Location
    st.markdown("#### 3. Career Goals & Location")
    career_goal = st.text_input(
        label="Target Career Goal / Aspiring Role *",
        placeholder="e.g., Machine Learning Engineer, Investment Analyst, Clinical Researcher",
        help="What specific job title, industry role, or career outcome are you aiming for?"
    )

    preferred_location = st.text_input(
        label="Preferred Country or Region (Optional)",
        placeholder="e.g., United States, Germany, Remote, London, or Flexible",
        help="Where would you ideally like to work? Leave blank if you are open to any location."
    )

    # Submission Button
    st.write("")
    submit_button = st.form_submit_button(
        label="🚀 Analyze My Profile",
        use_container_width=True,
        type="primary"
    )


# -----------------------------------------------------------------------------
# Report Presentation Helper
# -----------------------------------------------------------------------------
def display_career_report(report: CareerAnalysisReport, mode_label: str) -> None:
    """Renders the comprehensive career analysis report into structured Streamlit tabs."""
    # Executive Summary Banner
    st.markdown(f"### 💡 Advisor Summary\n{report.executive_summary}")
    st.write("")

    # Four Structured Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧭 Career Pathways (3)",
        "📊 Skills Gap Analysis",
        "📚 Learning Resources",
        "🗓️ 6-Month Roadmap"
    ])

    # Tab 1: Exactly 3 Career Recommendations
    with tab1:
        st.markdown(f"#### Top 3 Recommended Career Pathways ({mode_label})")
        st.caption("Realistic pathways tailored to your major, skills, and goals.")

        for idx, career in enumerate(report.career_recommendations, 1):
            with st.container(border=True):
                st.markdown(f"### Path {idx}: {career.role_title}")
                st.markdown(f"**Why this fits you:** {career.rationale}")

                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    st.markdown("**💼 Typical Responsibilities:**")
                    for resp in career.typical_responsibilities:
                        st.markdown(f"- {resp}")

                    st.markdown("**✅ Your Transferable Skills:**")
                    for skill in career.transferable_skills:
                        st.markdown(f"- {skill}")

                with c_col2:
                    st.markdown("**📈 Growth Areas to Cultivate:**")
                    for area in career.growth_areas:
                        st.markdown(f"- {area}")

    # Tab 2: Skills Gap Analysis
    with tab2:
        st.markdown("#### Comprehensive Skills Gap Analysis")

        gap_col1, gap_col2, gap_col3 = st.columns(3)
        with gap_col1:
            st.markdown("**💪 Current Strengths**")
            for s in report.skills_gap.identified_strengths:
                st.markdown(f"- {s}")

        with gap_col2:
            st.markdown("**🎯 Technical Gaps to Learn**")
            for t in report.skills_gap.technical_skills_to_acquire:
                st.markdown(f"- {t}")

        with gap_col3:
            st.markdown("**🤝 Soft Skills to Develop**")
            for soft in report.skills_gap.soft_skills_recommendations:
                st.markdown(f"- {soft}")

    # Tab 3: Learning Resources & Certifications
    with tab3:
        st.markdown("#### Recommended Certifications & Learning Resources")
        st.caption(
            "Note: Certifications build competencies and validate curiosity, "
            "but they do not guarantee employment."
        )

        for res in report.learning_resources:
            with st.container(border=True):
                st.markdown(f"**{res.name}**  `{res.resource_type}`")
                st.markdown(f"- **Provider / Platform:** {res.provider_or_platform}")
                st.markdown(f"- **Value for Your Journey:** {res.practical_value}")
                if getattr(res, "url", None):
                    st.markdown(f"- **Official Resource:** [{res.url}]({res.url})")

    # Tab 4: Sequential 6-Month Roadmap
    with tab4:
        st.markdown("#### Structured 6-Month Action Roadmap")
        st.caption("A structured step-by-step timeline broken into three 2-month phases.")

        for phase in report.six_month_roadmap:
            with st.container(border=True):
                st.markdown(f"### 🗓️ {phase.month_range}: {phase.phase_title}")

                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    st.markdown("**🎯 Key Milestones:**")
                    for m in phase.focus_milestones:
                        st.markdown(f"- {m}")

                with r_col2:
                    st.markdown("**📋 Weekly Action Items:**")
                    for a in phase.action_items:
                        st.markdown(f"- {a}")

    # Ethical Advisory Note at the bottom
    st.write("")
    st.info(f"⚖️ **Career Advisory Notice:** {report.advisory_note}", icon="ℹ️")


# -----------------------------------------------------------------------------
# Form Processing & Execution
# -----------------------------------------------------------------------------
if submit_button:
    # Validate required input fields
    is_valid, error_message = validate_student_input(major=major, career_goal=career_goal)

    if not is_valid:
        st.error(f"⚠️ **Incomplete Form:** {error_message}")
    else:
        # Clean and normalize inputs
        parsed_skills = parse_comma_separated_list(skills_raw)
        parsed_interests = parse_comma_separated_list(interests_raw)
        clean_location = preferred_location.strip() if preferred_location.strip() else "Flexible / Open to any location"
        formatted_skills = ", ".join(parsed_skills) if parsed_skills else "None specified"
        formatted_interests = ", ".join(parsed_interests) if parsed_interests else "None specified"

        profile = UserProfile(
            academic_background=major.strip(),
            study_level=study_level,
            skills=formatted_skills,
            interests=formatted_interests,
            goals=career_goal.strip(),
            preferred_location=clean_location
        )

        # Always Display Profile Summary First
        with st.expander("📋 Review Submitted Profile Summary", expanded=False):
            sum_col1, sum_col2 = st.columns(2)
            with sum_col1:
                st.markdown(f"**📚 Field of Study:** {profile.academic_background}")
                st.markdown(f"**🎓 Degree Level:** {profile.study_level}")
                st.markdown(f"**📍 Location Preference:** {profile.preferred_location}")
            with sum_col2:
                st.markdown(f"**🎯 Target Career Goal:** {profile.goals}")
                st.markdown(f"**🛠️ Skills:** {profile.skills}")
                st.markdown(f"**💡 Interests:** {profile.interests}")

        # Choose execution path: Gemini AI vs Local Guidance
        if has_gemini_key:
            try:
                with st.spinner(f"🤖 Consulting Google Gemini AI Advisor ({DEFAULT_GEMINI_MODEL})..."):
                    report = generate_career_guidance(
                        profile=profile,
                        api_key=active_api_key,
                        model=DEFAULT_GEMINI_MODEL
                    )

                st.info(f"✨ **Gemini AI Analysis:** Generated using Google Gemini (`{DEFAULT_GEMINI_MODEL}`).", icon="🤖")
                st.success("🎉 **Career Guidance Analysis Complete!**")
                display_career_report(report, mode_label="Gemini AI Analysis")

            except MissingAPIKeyError as err:
                st.warning(f"🔑 {err}", icon="⚠️")

            except AuthenticationAPIError as err:
                st.error(f"🚫 {err}", icon="❌")
                st.info("💡 You can clear or remove the invalid key to explore options using Local Career Guidance.", icon="ℹ️")

            except RateLimitAPIError as err:
                st.error(f"⏳ {err}", icon="💳")
                st.info(
                    "💡 Note: The Gemini API free tier has rate limits (RPM / RPD). "
                    "You can wait a moment before trying again or use Local Career Guidance.",
                    icon="ℹ️"
                )

            except NetworkAPIError as err:
                st.error(f"🌐 {err}", icon="📡")

            except InvalidResponseError as err:
                st.error(f"⚠️ {err}", icon="⚠️")

            except CareerGuidanceError as err:
                st.error(f"❌ {err}", icon="🚨")

        else:
            # Fallback mode when no API key is available
            st.info(
                "ℹ️ **Gemini AI is not configured. Showing local career guidance so you can still explore your options.**\n\n"
                "*(To enable AI-powered recommendations, add your `GEMINI_API_KEY` to `.streamlit/secrets.toml` or the sidebar.)*",
                icon="ℹ️"
            )

            with st.spinner("🔍 Generating rule-based local career guidance..."):
                report = generate_local_guidance(profile=profile)

            st.success("📋 **Local Career Guidance Analysis Complete!**")
            display_career_report(report, mode_label="Local Career Guidance")
