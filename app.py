"""
app.py — ClauseIQ: AI Legal Advisor for India
Streamlit frontend · Multi-agent pipeline powered by Groq + Llama 3.3 70B
"""

import io
import datetime
import streamlit as st
import PyPDF2

from agent import (
    analyze_document,
    simulate_scenario,
    parse_output,
    parse_negotiation_scripts,
    generate_negotiation_scripts,
    generate_consumer_motion,
    calculate_danger_score,
)

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ClauseIQ — AI Legal Advisor for India",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e2b96f; font-size: 2.5rem; margin: 0; }
    .main-header p  { color: #a0aec0; margin: 0.5rem 0 0 0; font-size: 1.1rem; }

    /* ── Legal Disclaimer Banner ── */
    .disclaimer-banner {
        background: #1c1207;
        border: 1px solid #92400e;
        border-left: 4px solid #d97706;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 1.5rem;
        font-size: 0.84rem;
        color: #fbbf24;
        line-height: 1.6;
    }
    .disclaimer-banner b { color: #f59e0b; }

    /* ── Doc type banner ── */
    .doc-type-banner {
        background: linear-gradient(135deg, #1e3a5f, #0f2744);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .doc-type-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #93c5fd;
        margin-bottom: 2px;
    }
    .doc-type-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2b96f;
    }
    .doc-meta-item {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* ── Danger Gauge ── */
    .gauge-container {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .gauge-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    .gauge-score {
        font-size: 5rem;
        font-weight: 900;
        line-height: 1;
        margin: 0.25rem 0;
    }
    .gauge-verdict {
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.5rem;
        padding: 4px 16px;
        border-radius: 20px;
        display: inline-block;
    }
    .score-safe    { color: #10b981; }
    .score-caution { color: #f59e0b; }
    .score-danger  { color: #ef4444; }
    .score-extreme { color: #dc2626; }
    .badge-safe    { background: #d1fae5; color: #065f46; }
    .badge-caution { background: #fef3cd; color: #92400e; }
    .badge-danger  { background: #fee2e2; color: #991b1b; }
    .badge-extreme { background: #7f1d1d; color: #fca5a5; }
    .gauge-bar-wrap {
        width: 100%;
        max-width: 400px;
        margin: 1rem auto 0 auto;
    }

    /* ── Decision boxes ── */
    .decision-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid;
    }
    .decision-sign   { background: #d1fae5; border-color: #059669; }
    .decision-review { background: #fef3cd; border-color: #d97706; }
    .decision-no     { background: #fee2e2; border-color: #dc2626; }

    /* ── Law citation ── */
    .law-citation {
        background: #1e3a5f;
        border-left: 3px solid #3b82f6;
        padding: 6px 12px;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #93c5fd;
        margin: 6px 0;
        font-style: italic;
    }

    /* ── Metric cards ── */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    .metric-number { font-size: 2rem; font-weight: 700; }
    .metric-label  { font-size: 0.8rem; color: #64748b; margin-top: 4px; }

    /* ── Pipeline steps ── */
    .pipeline-step {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #f8fafc;
        font-weight: 500;
    }

    /* ── Negotiation cards ── */
    .nego-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .nego-channel {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .nego-whatsapp { color: #25d366; }
    .nego-email    { color: #60a5fa; }
    .nego-message {
        background: #1e293b;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.88rem;
        color: #e2e8f0;
        white-space: pre-wrap;
        line-height: 1.6;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding: 1rem 0;
        border-top: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⚖️ ClauseIQ</h1>
    <p>Multi-Agent AI Legal Advisor · Built for India · Powered by Groq + Llama 3.3 70B</p>
</div>
""", unsafe_allow_html=True)

# ── Legal Disclaimer (always visible) ────────
st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <b>Legal Disclaimer:</b> ClauseIQ is an AI-powered tool for <b>informational purposes only</b>.
    It does <b>not</b> constitute legal advice, and no attorney-client relationship is created.
    Always consult a qualified Indian lawyer before signing any legal document or filing any complaint.
    Law citations are AI-generated and may not reflect the latest amendments — verify independently.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    language = st.selectbox("🌐 Output Language", ["English", "Hindi"])
    st.divider()

    st.markdown("### 🤖 Agent Pipeline")
    st.markdown("""
    <div class="pipeline-step">🗂️ <b>Agent 0</b> — Document Classifier</div>
    <div class="pipeline-step">🔍 <b>Agent 1</b> — Clause Extractor</div>
    <div class="pipeline-step">⚠️ <b>Agent 2</b> — Risk Evaluator</div>
    <div class="pipeline-step">💡 <b>Agent 3</b> — Legal Advisor + ⚖️ Law Citations</div>
    <div class="pipeline-step">🧠 <b>Decision Engine</b> — Danger Score</div>
    <div class="pipeline-step">✍️ <b>Agent 4</b> — Negotiation Scripts</div>
    <div class="pipeline-step">🏛️ <b>Agent 5</b> — Consumer Court Motion</div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📋 About ClauseIQ")
    st.markdown("""
ClauseIQ uses a **6-agent AI pipeline** to:
- Auto-detect your document type
- Extract and evaluate every clause
- Cite exact Indian laws
- Compute a 0–100 Danger Score
- Write negotiation messages to send immediately
- Draft a Consumer Court complaint (CPA 2019)

**Model:** Groq — Llama 3.3 70B Versatile

**Free API key:** [console.groq.com](https://console.groq.com)
    """)
    st.divider()
    st.caption("Built with ❤️ for everyday Indians · v1.0")


# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📄 Upload Your Document")
    uploaded_file = st.file_uploader(
        "Upload any contract, agreement, or legal document",
        type=["pdf", "txt"],
        help="Supports PDF (with selectable text) and TXT files. Recommended max: 10MB.",
    )

with col2:
    st.markdown("### 🎯 Supported Document Types")
    st.markdown("""
- 🏠 Rental Agreements
- 💼 Employment Contracts
- 🎓 College Bonding Letters
- 🤝 Internship Offers
- 🏦 Loan Documents
- 🔒 NDAs & Service Agreements
    """)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def extract_text_from_file(file) -> str:
    """
    Extract text from an uploaded PDF or TXT file.
    Raises RuntimeError with user-friendly messages on failure.
    """
    if file.type == "application/pdf":
        try:
            raw_bytes = file.read()
            if len(raw_bytes) == 0:
                raise RuntimeError("📄 The uploaded PDF is empty (0 bytes).")
            reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
            if len(reader.pages) == 0:
                raise RuntimeError("📄 The PDF has no pages.")
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            if not text.strip():
                raise RuntimeError(
                    "📄 **No text could be extracted from this PDF.**\n\n"
                    "This usually means the PDF is a scanned image. "
                    "Try converting it with an OCR tool (e.g. Adobe Acrobat, Smallpdf) first, "
                    "or copy-paste the text into a .txt file and upload that."
                )
            return text
        except RuntimeError:
            raise
        except PyPDF2.errors.PdfReadError as e:
            raise RuntimeError(
                f"📄 **Could not read the PDF** — the file may be corrupted or password-protected.\n"
                f"Error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"📄 **Unexpected error reading PDF:** {e}") from e

    # TXT / plain text
    try:
        raw = file.read()
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("latin-1")
        except Exception as e:
            raise RuntimeError(
                f"📄 **Could not decode the text file.** "
                f"Please save it as UTF-8 and re-upload. Error: {e}"
            ) from e
    except Exception as e:
        raise RuntimeError(f"📄 **Unexpected error reading file:** {e}") from e


def get_score_style(score: int) -> tuple[str, str, str]:
    """Returns (score_css_class, badge_css_class, verdict_text)."""
    if score <= 30:
        return "score-safe",    "badge-safe",    "LOW RISK — Generally Safe to Sign"
    elif score <= 55:
        return "score-caution", "badge-caution", "MODERATE RISK — Review Before Signing"
    elif score <= 75:
        return "score-danger",  "badge-danger",  "HIGH RISK — Negotiate First"
    else:
        return "score-extreme", "badge-extreme", "EXTREME RISK — Do Not Sign As-Is"


def render_danger_gauge(score: int) -> None:
    """Render the animated danger score gauge card."""
    score_cls, badge_cls, verdict = get_score_style(score)
    bar_color = {
        "score-safe":    "#10b981",
        "score-caution": "#f59e0b",
        "score-danger":  "#ef4444",
        "score-extreme": "#dc2626",
    }[score_cls]
    fill_pct = score

    st.markdown(f"""
    <div class="gauge-container">
        <div class="gauge-label">📊 Document Danger Score</div>
        <div class="gauge-score {score_cls}">{score}</div>
        <div style="color:#475569; font-size:0.8rem;">out of 100</div>
        <span class="gauge-verdict {badge_cls}">{verdict}</span>
        <div class="gauge-bar-wrap">
            <svg viewBox="0 0 400 18" xmlns="http://www.w3.org/2000/svg" style="width:100%;margin-top:12px;">
                <rect x="0" y="4" width="400" height="10" rx="5" fill="#1e293b"/>
                <rect x="0" y="4" width="{fill_pct * 4}" height="10" rx="5" fill="{bar_color}"/>
                <rect x="100" y="2" width="2" height="14" fill="#334155"/>
                <rect x="200" y="2" width="2" height="14" fill="#334155"/>
                <rect x="300" y="2" width="2" height="14" fill="#334155"/>
                <text x="0"   y="17" fill="#475569" font-size="7">0</text>
                <text x="97"  y="17" fill="#475569" font-size="7">25</text>
                <text x="197" y="17" fill="#475569" font-size="7">50</text>
                <text x="297" y="17" fill="#475569" font-size="7">75</text>
                <text x="390" y="17" fill="#475569" font-size="7">100</text>
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_doc_type_banner(doc_meta: dict) -> None:
    doc_type     = doc_meta.get("DOC_TYPE", "Legal Document")
    parties      = doc_meta.get("PARTIES", "—")
    jurisdiction = doc_meta.get("JURISDICTION", "India")
    summary      = doc_meta.get("SUMMARY", "")

    type_icons = {
        "Rental Agreement": "🏠", "Employment Contract": "💼",
        "Internship Offer": "🤝", "College Bonding Letter": "🎓",
        "Loan Agreement": "🏦", "Non-Disclosure Agreement": "🔒",
        "Service Agreement": "📱", "Freelance Contract": "💻",
    }
    icon = type_icons.get(doc_type, "📄")

    st.markdown(f"""
    <div class="doc-type-banner">
        <div style="font-size:2.5rem">{icon}</div>
        <div style="flex:1">
            <div class="doc-type-label">Auto-Detected Document Type</div>
            <div class="doc-type-value">{doc_type}</div>
            <div class="doc-meta-item">👥 {parties} &nbsp;·&nbsp; 📍 {jurisdiction}</div>
            {f'<div class="doc-meta-item" style="margin-top:4px;color:#cbd5e1">{summary}</div>' if summary else ""}
        </div>
        <div style="background:#1e3a5f;border:1px solid #3b82f6;border-radius:8px;padding:6px 12px;font-size:0.75rem;color:#93c5fd;white-space:nowrap;">
            ✅ Context-aware<br>risk analysis
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_clauses(clauses: list, language: str) -> None:
    if not clauses:
        st.warning(
            "⚠️ No clauses were parsed from the agent output. "
            "Check the **Agent Internals** tab to see the raw output."
        )
        return

    high   = [c for c in clauses if c.get("RISK") == "HIGH"]
    medium = [c for c in clauses if c.get("RISK") == "MEDIUM"]
    low    = [c for c in clauses if c.get("RISK") == "LOW"]

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-number" style="color:#dc2626">{len(high)}</div>
            <div class="metric-label">High Risk Clauses</div>
        </div>
        <div class="metric-card">
            <div class="metric-number" style="color:#d97706">{len(medium)}</div>
            <div class="metric-label">Medium Risk Clauses</div>
        </div>
        <div class="metric-card">
            <div class="metric-number" style="color:#059669">{len(low)}</div>
            <div class="metric-label">Low Risk Clauses</div>
        </div>
        <div class="metric-card">
            <div class="metric-number" style="color:#3b82f6">{len(clauses)}</div>
            <div class="metric-label">Total Clauses Found</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if high:
        st.markdown("#### 🔴 High Risk Clauses — Must Review Before Signing")
        for clause in high:
            score_label = f" · Risk Score: {clause['RISK_SCORE']}/10" if clause.get("RISK_SCORE") else ""
            with st.expander(f"🔴 {clause.get('CLAUSE', 'Unknown Clause')}{score_label}", expanded=True):
                st.markdown(f"**⚡ Why this is risky:** {clause.get('WHY THIS IS RISKY', 'N/A')}")
                st.markdown(f"**📖 What it means:** {clause.get('WHAT IT MEANS', 'N/A')}")
                if clause.get("INDIAN_LAW"):
                    st.markdown(f'<div class="law-citation">⚖️ Indian Law: {clause["INDIAN_LAW"]}</div>', unsafe_allow_html=True)
                st.markdown(f"**✅ What to do:** {clause.get('SUGGESTION', 'N/A')}")

    if medium:
        st.markdown("#### 🟡 Medium Risk Clauses — Review Recommended")
        for clause in medium:
            score_label = f" · Risk Score: {clause['RISK_SCORE']}/10" if clause.get("RISK_SCORE") else ""
            with st.expander(f"🟡 {clause.get('CLAUSE', 'Unknown Clause')}{score_label}"):
                st.markdown(f"**⚡ Why this is risky:** {clause.get('WHY THIS IS RISKY', 'N/A')}")
                st.markdown(f"**📖 What it means:** {clause.get('WHAT IT MEANS', 'N/A')}")
                if clause.get("INDIAN_LAW"):
                    st.markdown(f'<div class="law-citation">⚖️ Indian Law: {clause["INDIAN_LAW"]}</div>', unsafe_allow_html=True)
                st.markdown(f"**✅ What to do:** {clause.get('SUGGESTION', 'N/A')}")

    if low:
        st.markdown("#### 🟢 Low Risk Clauses — Generally Safe")
        for clause in low:
            with st.expander(f"🟢 {clause.get('CLAUSE', 'Unknown Clause')}"):
                st.markdown(f"**📖 What it means:** {clause.get('WHAT IT MEANS', 'N/A')}")
                if clause.get("INDIAN_LAW"):
                    st.markdown(f'<div class="law-citation">⚖️ Indian Law: {clause["INDIAN_LAW"]}</div>', unsafe_allow_html=True)
                st.markdown(f"**✅ Note:** {clause.get('SUGGESTION', 'N/A')}")


def display_final_decision(decision: dict, danger_score: int) -> None:
    verdict      = decision.get("FINAL_DECISION", "REVIEW NEEDED")
    overall_risk = decision.get("OVERALL_RISK", "MEDIUM")
    reason       = decision.get("REASON", "")
    key_clauses  = decision.get("KEY_CLAUSES_TO_NEGOTIATE", "")

    st.markdown("---")
    st.markdown("## 🧠 Decision Engine — Final Call")

    render_danger_gauge(danger_score)

    if verdict == "SIGN":
        st.markdown(f"""
        <div class="decision-box decision-sign">
            <h2 style="color:#065f46;margin:0">✅ VERDICT: SAFE TO SIGN</h2>
            <p style="color:#047857;margin:0.5rem 0 0 0">Overall Risk: {overall_risk}</p>
        </div>""", unsafe_allow_html=True)
    elif verdict == "SIGN WITH CHANGES":
        st.markdown(f"""
        <div class="decision-box decision-review">
            <h2 style="color:#92400e;margin:0">⚠️ VERDICT: SIGN WITH CHANGES</h2>
            <p style="color:#b45309;margin:0.5rem 0 0 0">Overall Risk: {overall_risk}</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="decision-box decision-no">
            <h2 style="color:#991b1b;margin:0">🚫 VERDICT: DO NOT SIGN</h2>
            <p style="color:#dc2626;margin:0.5rem 0 0 0">Overall Risk: {overall_risk}</p>
        </div>""", unsafe_allow_html=True)

    if reason:
        st.markdown(f"**🧠 Reasoning:** {reason}")
    if key_clauses:
        st.markdown(f"**🔑 Key clauses to negotiate:** {key_clauses}")


def display_negotiation_scripts(scripts: list) -> None:
    if not scripts:
        st.info("No negotiation scripts could be parsed. Check the Agent Internals tab for raw output.")
        return

    for script in scripts:
        clause_name = script.get("CLAUSE", "Clause")
        st.markdown(f"### ✍️ {clause_name}")

        col_wa, col_em = st.columns(2)

        with col_wa:
            st.markdown('<div class="nego-channel nego-whatsapp">💬 WhatsApp Message</div>', unsafe_allow_html=True)
            msg = script.get("WHATSAPP_MESSAGE", "—")
            st.markdown(f'<div class="nego-message">{msg}</div>', unsafe_allow_html=True)
            st.code(msg, language=None)

        with col_em:
            subject = script.get("EMAIL_SUBJECT", "")
            body    = script.get("EMAIL_BODY", "—")
            st.markdown('<div class="nego-channel nego-email">📧 Email</div>', unsafe_allow_html=True)
            if subject:
                st.markdown(f"**Subject:** `{subject}`")
            st.markdown(f'<div class="nego-message">{body}</div>', unsafe_allow_html=True)
            st.code(body, language=None)

        st.divider()


def build_report_markdown(
    doc_meta: dict,
    clauses: list,
    final_decision: dict,
    danger_score: int,
    filename: str,
) -> str:
    """Build a Markdown report of the full analysis for download."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# ClauseIQ Analysis Report",
        f"**File:** {filename}  |  **Generated:** {ts}",
        "",
        "---",
        "",
        "## Document Overview",
        f"- **Type:** {doc_meta.get('DOC_TYPE', 'Unknown')}",
        f"- **Parties:** {doc_meta.get('PARTIES', 'Unknown')}",
        f"- **Jurisdiction:** {doc_meta.get('JURISDICTION', 'India')}",
        f"- **Summary:** {doc_meta.get('SUMMARY', '')}",
        "",
        "---",
        "",
        f"## Danger Score: {danger_score}/100",
        f"**Verdict:** {final_decision.get('FINAL_DECISION', 'N/A')}  |  "
        f"**Overall Risk:** {final_decision.get('OVERALL_RISK', 'N/A')}",
        "",
        f"**Reason:** {final_decision.get('REASON', '')}",
        f"**Clauses to negotiate:** {final_decision.get('KEY_CLAUSES_TO_NEGOTIATE', '')}",
        "",
        "---",
        "",
        "## Clause Analysis",
        "",
    ]

    for clause in clauses:
        risk = clause.get("RISK", "LOW")
        emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk, "⚪")
        lines += [
            f"### {emoji} {clause.get('CLAUSE', 'Unknown')} ({risk})",
            f"- **Risk Score:** {clause.get('RISK_SCORE', 'N/A')}/10",
            f"- **Why risky:** {clause.get('WHY THIS IS RISKY', 'N/A')}",
            f"- **What it means:** {clause.get('WHAT IT MEANS', 'N/A')}",
            f"- **Indian Law:** {clause.get('INDIAN_LAW', 'N/A')}",
            f"- **What to do:** {clause.get('SUGGESTION', 'N/A')}",
            "",
        ]

    lines += [
        "---",
        "",
        "> **Disclaimer:** This report is AI-generated for informational purposes only.",
        "> It does not constitute legal advice. Consult a qualified lawyer before signing.",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN ANALYSIS FLOW
# ─────────────────────────────────────────────
if uploaded_file:
    # Extract text from file
    try:
        with st.spinner("📖 Reading document..."):
            doc_text = extract_text_from_file(uploaded_file)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    word_count = len(doc_text.split())
    char_count = len(doc_text)
    st.success(f"✅ Document loaded — **{word_count:,} words** | {char_count:,} characters | `{uploaded_file.name}`")
    st.divider()

    if st.button("🔍 Run Full Agent Analysis", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="🗂️ Agent 0 — Detecting document type...")

        try:
            with st.spinner("⚙️ Running 5-agent pipeline — this takes 15–30 seconds..."):
                result = analyze_document(doc_text, language)

            progress_bar.progress(100, text="✅ All agents complete!")

            # Persist to session state
            st.session_state["result"]      = result
            st.session_state["doc_text"]    = doc_text
            st.session_state["language"]    = language
            st.session_state["filename"]    = uploaded_file.name
            # Clear stale downstream state on new analysis
            for key in ["nego_scripts", "nego_raw", "scenario", "motion_draft"]:
                st.session_state.pop(key, None)

        except RuntimeError as e:
            progress_bar.empty()
            st.error(str(e))
            st.stop()

    # ── Display results ──────────────────────────────────────────────
    if "result" in st.session_state:
        result       = st.session_state["result"]
        doc_meta     = result.get("doc_meta", {})
        doc_type     = result.get("doc_type", "Legal Document")
        clauses, final_decision = parse_output(result["agent3_final"])
        danger_score = calculate_danger_score(clauses, final_decision)

        render_doc_type_banner(doc_meta)

        # ── Download Report Button ───────────────────────────────────
        report_md = build_report_markdown(
            doc_meta, clauses, final_decision, danger_score,
            st.session_state.get("filename", "document")
        )
        st.download_button(
            label="📥 Download Analysis Report (.md)",
            data=report_md.encode("utf-8"),
            file_name=f"ClauseIQ_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.divider()

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Clause Analysis",
            "🧠 Decision + Danger Score",
            "✍️ Negotiation Scripts",
            "⚡ Risk Summary",
            "🏛️ Consumer Court Motion",
            "🔬 Agent Internals",
        ])

        with tab1:
            st.markdown("### 📊 Clause-by-Clause Analysis")
            display_clauses(clauses, st.session_state["language"])

        with tab2:
            display_final_decision(final_decision, danger_score)

        with tab3:
            st.markdown("### ✍️ Negotiation Scripts — Ready to Send")
            st.markdown(
                "Agent 4 writes the **exact WhatsApp message and email** you can send to your "
                f"counterparty to negotiate every risky clause in your **{doc_type}**."
            )

            high_medium = [c for c in clauses if c.get("RISK") in ("HIGH", "MEDIUM")]
            if not high_medium:
                st.success("🎉 No high or medium risk clauses — nothing to negotiate!")
            else:
                st.info(f"Found **{len(high_medium)} clauses** to negotiate. Click below to generate scripts.")

                if st.button("✍️ Generate Negotiation Scripts", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Agent 4 writing negotiation messages..."):
                            raw_scripts = generate_negotiation_scripts(
                                clauses,
                                doc_type,
                                st.session_state["language"],
                            )
                        st.session_state["nego_scripts"] = parse_negotiation_scripts(raw_scripts)
                        st.session_state["nego_raw"]     = raw_scripts
                    except RuntimeError as e:
                        st.error(str(e))

                if "nego_scripts" in st.session_state:
                    display_negotiation_scripts(st.session_state["nego_scripts"])

        with tab4:
            st.markdown("### ⚡ Instant Risk Summary")
            st.markdown("**What happens if you sign this document right now, without negotiating anything?**")

            if st.button("🤔 Show Risk Summary", use_container_width=True):
                try:
                    with st.spinner("Analysing worst-case risks..."):
                        scenario = simulate_scenario(
                            st.session_state["doc_text"],
                            st.session_state["language"],
                        )
                    st.session_state["scenario"] = scenario
                except RuntimeError as e:
                    st.error(str(e))

            if "scenario" in st.session_state:
                st.markdown("#### ⚠️ Risk Summary — If You Sign As-Is")
                st.markdown(st.session_state["scenario"])

        with tab5:
            st.markdown("### 🏛️ Consumer Court Motion — File Your Own Case")
            st.info(
                "Under **Section 35 of the Consumer Protection Act, 2019** and **Order III CPC**, "
                "any person can represent themselves before the District Consumer "
                "Disputes Redressal Commission (DCDRC). No lawyer required."
            )

            with st.form("consumer_court_form"):
                st.markdown("#### 📝 Enter Your Details")
                col_a, col_b = st.columns(2)
                with col_a:
                    complainant_name    = st.text_input("Your Full Name *")
                    complainant_address = st.text_area("Your Full Address *", height=80)
                    district            = st.text_input("District (for Forum) *", value="Pune")
                    state               = st.text_input("State *", value="Maharashtra")
                with col_b:
                    opposite_party         = st.text_input("Opposite Party Name (Company/Person) *")
                    opposite_party_address = st.text_area("Opposite Party Address *", height=80)
                    relief_amount          = st.text_input("Amount of Relief Claimed (₹) *", placeholder="50000")

                grievance_summary = st.text_area(
                    "Describe Your Grievance in Brief *",
                    placeholder="e.g. The landlord unlawfully withheld security deposit of ₹50,000 despite no damage to property...",
                    height=100,
                )

                submitted = st.form_submit_button(
                    "🏛️ Generate Consumer Court Motion",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                required = [complainant_name, complainant_address, opposite_party,
                            grievance_summary, relief_amount, district, state]
                if not all(f.strip() for f in required):
                    st.error("⚠️ Please fill in **all required fields (*)** before generating.")
                else:
                    try:
                        with st.spinner("Agent 5 drafting your Consumer Court complaint..."):
                            motion_draft = generate_consumer_motion(
                                doc_text=st.session_state["doc_text"],
                                clauses=clauses,
                                complainant_name=complainant_name,
                                complainant_address=complainant_address,
                                opposite_party=opposite_party,
                                opposite_party_address=opposite_party_address,
                                grievance_summary=grievance_summary,
                                relief_amount=relief_amount,
                                district=district,
                                state=state,
                                language=st.session_state["language"],
                            )
                        st.session_state["motion_draft"] = motion_draft
                    except RuntimeError as e:
                        st.error(str(e))

            if "motion_draft" in st.session_state:
                st.markdown("---")
                st.markdown("#### 📄 Your Consumer Court Complaint Draft")
                st.markdown(
                    "_Review carefully before filing. Fill in blanks marked \\____. "
                    "Print on stamp paper if required by your District Forum._"
                )
                motion_text = st.session_state["motion_draft"]
                st.text_area("Motion Draft (copy to edit)", motion_text, height=500)
                st.download_button(
                    label="📥 Download Complaint Draft (.txt)",
                    data=motion_text.encode("utf-8"),
                    file_name=f"Consumer_Court_Complaint_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        with tab6:
            st.markdown("### 🔬 Agent Internals — Reasoning Transparency")
            st.caption("Raw output from each agent in the pipeline. Useful for debugging or verifying the AI's reasoning.")

            with st.expander("🗂️ Agent 0 — Document Classification"):
                st.text(result.get("doc_meta_raw", "(empty)"))

            with st.expander("🔍 Agent 1 — Raw Clause Extraction"):
                st.text(result.get("agent1_clauses", "(empty)"))

            with st.expander("⚠️ Agent 2 — Risk Evaluation"):
                st.text(result.get("agent2_risks", "(empty)"))

            with st.expander("💡 Agent 3 — Legal Advice + Law Citations + Decision"):
                st.text(result.get("agent3_final", "(empty)"))

            if "nego_raw" in st.session_state:
                with st.expander("✍️ Agent 4 — Raw Negotiation Scripts"):
                    st.text(st.session_state["nego_raw"])

else:
    # ── Landing state ────────────────────────────────────────────────
    st.divider()
    st.markdown("### 👆 Upload a document above to get started")
    st.markdown("""
**ClauseIQ will:**
1. 🗂️ Auto-detect document type (Rental / Employment / Loan etc.) — *Agent 0*
2. 🔍 Extract every clause with context-aware analysis — *Agent 1*
3. ⚠️ Score each clause HIGH / MEDIUM / LOW + Risk Score /10 — *Agent 2*
4. 💡 Explain each clause in plain language + cite Indian laws — *Agent 3*
5. 🧠 Compute a 0–100 Danger Score and make SIGN / DO NOT SIGN decision
6. ✍️ Write ready-to-send WhatsApp & Email negotiation scripts — *Agent 4*
7. 🏛️ Draft a Consumer Court motion (CPA 2019) to file your own case — *Agent 5*

**Try uploading:** A rental agreement, internship offer letter, employment contract, or college bonding document.
    """)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    ⚖️ ClauseIQ · AI Legal Advisor for India · Powered by Groq + Llama 3.3 70B<br>
    <b>Disclaimer:</b> For informational purposes only. Not a substitute for professional legal advice.<br>
    © 2024 · Built with Streamlit
</div>
""", unsafe_allow_html=True)
