"""
agent.py — ClauseIQ Multi-Agent Backend
Groq Llama 3.1 8B (fast) + Llama 3.3 70B (heavy) · 6-Agent Pipeline for Indian Legal Documents

Agents:
  0 — Document Type Detector          → llama-3.1-8b-instant   (fast, separate quota)
  1 — Clause Extractor (context-aware) → llama-3.1-8b-instant   (fast, separate quota)
  2 — Risk Evaluator (context-aware)   → llama-3.3-70b-versatile
  3 — Legal Advisor + Indian Law Citations + Decision Engine → llama-3.3-70b-versatile
  4 — Negotiation Script Generator (WhatsApp & Email)        → llama-3.3-70b-versatile
  5 — Consumer Court Motion Drafter (CPA 2019)               → llama-3.3-70b-versatile

Rate-limit strategy (Groq free tier):
  - Agents 0 & 1 use llama-3.1-8b-instant (~14,400 tok/min) — separate bucket from 70B
  - Agents 2–5 use llama-3.3-70b-versatile (~6,000 tok/min)
  - 8-second sleep between every agent call lets counters partially reset
  - Total pipeline: ~40s on free tier instead of crashing
"""

import time
import streamlit as st
from groq import Groq

# ─────────────────────────────────────────────
# CLIENT INITIALISATION
# API key is loaded from Streamlit secrets.
# To run locally: create .streamlit/secrets.toml with:
#   GROQ_API_KEY = "your_key_here"
# Get a free key at: https://console.groq.com
# ─────────────────────────────────────────────

def _get_groq_client() -> Groq:
    """Return a Groq client, raising a clear error if the key is missing."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        raise RuntimeError(
            "🔑 **GROQ_API_KEY not found.**\n\n"
            "**Local development:** Add it to `.streamlit/secrets.toml`:\n"
            "```\nGROQ_API_KEY = \"your_key_here\"\n```\n\n"
            "**Streamlit Cloud:** Go to App Settings → Secrets and add the key.\n\n"
            "Get a free key at https://console.groq.com"
        )

    if not api_key or not api_key.strip():
        raise RuntimeError(
            "🔑 **GROQ_API_KEY is empty.** "
            "Please set a valid key in your Streamlit secrets."
        )

    return Groq(api_key=api_key.strip())


# ─────────────────────────────────────────────
# MODEL SELECTION
# Two separate Groq rate-limit buckets on free tier:
#   - 8B instant:      ~14,400 tokens/min  → Agents 0, 1
#   - 70B versatile:    ~6,000 tokens/min  → Agents 2, 3, 4, 5
# ─────────────────────────────────────────────
MODEL_FAST  = "llama-3.1-8b-instant"      # Agents 0, 1
MODEL_HEAVY = "llama-3.3-70b-versatile"   # Agents 2, 3, 4, 5

# Seconds to sleep between agent calls so Groq token counters can reset
_INTER_AGENT_DELAY = 8


# ─────────────────────────────────────────────
# AGENT 0 — Document Type Detector
# ─────────────────────────────────────────────
DOC_TYPE_PROMPT = """You are a legal document classifier for Indian legal documents.
Read the document and determine its type.

Output EXACTLY this format (nothing else):

DOC_TYPE: [Rental Agreement / Employment Contract / Internship Offer / College Bonding Letter / Loan Agreement / Service Agreement / Non-Disclosure Agreement / Freelance Contract / Other]
PARTIES: [Who are the two parties? e.g. "Landlord and Tenant", "Employer and Employee"]
JURISDICTION: [State/City if mentioned, else "India (General)"]
SUMMARY: [One sentence — what is this document about?]"""


# ─────────────────────────────────────────────
# AGENT 1 — Clause Analyzer (context-aware)
# ─────────────────────────────────────────────
def build_clause_analyzer_prompt(doc_type: str) -> str:
    context_hints = {
        "Rental Agreement":       "Pay special attention to: lock-in period, maintenance liability, security deposit terms, eviction clauses, rent escalation.",
        "Employment Contract":    "Pay special attention to: notice period, non-compete, IP ownership, probation terms, termination without cause, salary revision.",
        "Internship Offer":       "Pay special attention to: unpaid clauses, IP assignment, non-disclosure, extension clauses, conversion to full-time terms.",
        "College Bonding Letter": "Pay special attention to: penalty amounts, bond period, conditions for breaking bond, enforceability under Indian law.",
        "Loan Agreement":         "Pay special attention to: interest rate, prepayment penalty, default triggers, collateral forfeiture, cross-default.",
        "Non-Disclosure Agreement": "Pay special attention to: scope of confidential info, duration, one-sided vs mutual obligations, remedies clause.",
    }
    hint = context_hints.get(
        doc_type,
        "Pay attention to all clauses that limit rights, impose penalties, or favor one party."
    )

    return f"""You are Agent 1: Clause Analyzer.
This document has been identified as a: {doc_type}
{hint}

Your ONLY job is to extract every significant clause from the legal document.
For each clause output EXACTLY this format:

CLAUSE: [Name of clause]
TYPE: [Payment / Termination / Liability / Confidentiality / Penalty / Rights / Dispute / Other]
TEXT: [The exact or paraphrased clause content in 1-2 lines]
===

Extract ALL clauses. Do not skip any. Do not add analysis yet."""


# ─────────────────────────────────────────────
# AGENT 2 — Risk Evaluator (context-aware)
# ─────────────────────────────────────────────
def build_risk_evaluator_prompt(doc_type: str) -> str:
    return f"""You are Agent 2: Risk Evaluator.
You receive a list of extracted clauses from a {doc_type}.
For each clause evaluate risk specifically for an Indian citizen signing this type of document.

Output EXACTLY this format:

CLAUSE: [Same clause name from input]
RISK: [HIGH / MEDIUM / LOW]
RISK_SCORE: [A number from 1-10, where 10 is most dangerous]
WHY THIS IS RISKY: [1 line — specific legal or practical reason, considering this is a {doc_type}]
===

Context for {doc_type}: A clause that is normal in one contract type may be high-risk in another.
For example: a 3-month lock-in is standard in a Rental Agreement (LOW risk) but highly restrictive in an Employment Contract (HIGH risk).
Be strict. Clauses that limit rights, impose penalties, restrict exits, or favor one party heavily = HIGH risk."""


# ─────────────────────────────────────────────
# AGENT 3 — Legal Advisor + Indian Law Citations
# ─────────────────────────────────────────────
def build_advisor_prompt(language: str, doc_type: str) -> str:
    lang_instruction = (
        "Explain everything in simple Hindi. Keep legal terms in English but explain them."
        if language == "Hindi"
        else "Explain in simple plain English. No legal jargon."
    )

    law_examples = {
        "Rental Agreement":       "Transfer of Property Act 1882, Rent Control Acts (state-specific), Registration Act 1908",
        "Employment Contract":    "Industrial Disputes Act 1947, Shops and Establishments Act, Payment of Gratuity Act 1972, Contract Labour Act",
        "College Bonding Letter": "Indian Contract Act 1872 (Section 27 — restraint of trade), Constitution Article 19(1)(g)",
        "Loan Agreement":         "SARFAESI Act 2002, RBI Guidelines, Indian Contract Act 1872",
        "Non-Disclosure Agreement": "Indian Contract Act 1872, Information Technology Act 2000",
        "Internship Offer":       "Indian Contract Act 1872, Apprentices Act 1961, state labour laws",
    }
    law_hint = law_examples.get(doc_type, "Indian Contract Act 1872, relevant state laws")

    return f"""You are Agent 3: Legal Advisor with expertise in Indian law.
You receive clauses with their risk levels from a {doc_type}.
Provide practical advice for an everyday Indian person.
{lang_instruction}

Relevant laws to cite where applicable: {law_hint}

Output EXACTLY this format:

CLAUSE: [Same clause name]
RISK: [Same risk level]
RISK_SCORE: [Same score from Agent 2]
WHY THIS IS RISKY: [Same reason from Agent 2]
WHAT IT MEANS: [Plain language explanation — what does this actually mean for you? 1-2 sentences]
INDIAN_LAW: [Cite the specific Indian law, section, or act that applies. E.g. "Section 108, Transfer of Property Act 1882". If no specific law, write "Indian Contract Act 1872 (general enforceability)"]
SUGGESTION: [Exactly what the person should say, ask, or do before signing]
===

After ALL clauses, output the FINAL DECISION ENGINE block:

FINAL_DECISION_START
OVERALL_RISK: [LOW / MEDIUM / HIGH]
DANGER_SCORE: [A number from 0-100. Formula: weighted average of clause risk scores. 0=perfectly safe, 100=do not sign under any circumstance]
FINAL_DECISION: [SIGN / SIGN WITH CHANGES / DO NOT SIGN]
REASON: [2 lines — why this decision was made based on the clauses found]
KEY_CLAUSES_TO_NEGOTIATE: [Comma separated list of the most important clauses to fix]
FINAL_DECISION_END"""


# ─────────────────────────────────────────────
# AGENT 4 — Negotiation Script Generator
# ─────────────────────────────────────────────
def build_negotiation_prompt(language: str, doc_type: str) -> str:
    lang_instruction = (
        "Write the messages in Hindi (formal but simple)."
        if language == "Hindi"
        else "Write the messages in professional but simple English."
    )

    counterparty = {
        "Rental Agreement":       "landlord",
        "Employment Contract":    "HR / employer",
        "College Bonding Letter": "college administration",
        "Internship Offer":       "internship coordinator / HR",
        "Loan Agreement":         "bank representative / lender",
        "Non-Disclosure Agreement": "the other party's representative",
    }.get(doc_type, "the other party")

    return f"""You are Agent 4: Negotiation Script Writer.
You are given a list of HIGH and MEDIUM risk clauses from a {doc_type}.
Your job is to write ready-to-send negotiation messages that a regular Indian person can use immediately.
{lang_instruction}

For each HIGH or MEDIUM risk clause, write:

CLAUSE: [Clause name]
WHATSAPP_MESSAGE: [A short, polite WhatsApp message (3-5 lines) the person can send to their {counterparty} asking to change or clarify this clause. Sound human, not robotic. Start with "Hi," or "Namaste,"]
EMAIL_SUBJECT: [A professional email subject line]
EMAIL_BODY: [A 5-8 line professional email requesting the change. Include a specific proposed alternative where possible.]
===

Be tactful. The goal is to negotiate, not to threaten. The person still wants a good relationship with their {counterparty}."""


# ─────────────────────────────────────────────
# AGENT 5 — Consumer Court Motion Drafter
# ─────────────────────────────────────────────
CONSUMER_MOTION_PROMPT = """You are Agent 5: Consumer Court Motion Drafter.
You help Indian citizens file complaints before the District Consumer Disputes Redressal Commission (DCDRC) under the Consumer Protection Act 2019.

The complainant is representing themselves (allowed under Order III CPC and Section 35 of CPA 2019).

Using the document text, clause analysis, and complainant details provided, generate a FORMAL complaint draft with ALL these sections:

---
BEFORE THE DISTRICT CONSUMER DISPUTES REDRESSAL COMMISSION
[District], [State]

COMPLAINT CASE NO.: ____/20__

IN THE MATTER OF:
[Complainant Name], [Address] -- COMPLAINANT

VERSUS

[Opposite Party Name and address] -- OPPOSITE PARTY
---

COMPLAINT UNDER SECTION 35 OF THE CONSUMER PROTECTION ACT, 2019

1. JURISDICTION AND MAINTAINABILITY
[Why this forum has jurisdiction -- pecuniary and territorial]

2. FACTS OF THE CASE
[Numbered paragraphs describing events chronologically]

3. CAUSE OF ACTION
[When and how the cause of action arose]

4. UNFAIR TRADE PRACTICE / DEFICIENCY IN SERVICE
[Specific clauses that constitute deficiency under CPA 2019 Section 2(11) or unfair trade practice under Section 2(47)]

5. LEGAL GROUNDS
[Cite: Consumer Protection Act 2019 Sections 2(7), 2(11), 2(47), 35, 39; Indian Contract Act 1872 relevant sections; any other applicable law]

6. RELIEF SOUGHT
[Specific reliefs: refund, compensation, penalty, direction to opposite party]

7. DECLARATION
I, the complainant, do hereby declare that the facts stated above are true and correct to the best of my knowledge and belief.

Date: ____
Place: ____
Complainant (Self-represented)
Signature: ____

---

OUTPUT ONLY the complete complaint draft. Use formal legal language but keep it clear."""


# ─────────────────────────────────────────────
# CORE API RUNNER
# ─────────────────────────────────────────────
def run_agent(
    system_prompt: str,
    user_content: str,
    retries: int = 2,
    max_tokens: int = 2000,
    model: str = MODEL_HEAVY,
) -> str:
    """
    Call Groq API with a system + user message.
    - model: pass MODEL_FAST for Agents 0/1, MODEL_HEAVY for Agents 2-5
    - Retries on transient errors with exponential backoff.
    - Raises RuntimeError with user-friendly messages on failure.
    """
    client = _get_groq_client()
    last_error = None

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
            )
            return response.choices[0].message.content

        except Exception as e:
            last_error = e
            err_str = str(e).lower()

            # Auth errors — no point retrying
            if any(k in err_str for k in ["invalid_api_key", "authentication", "401"]):
                raise RuntimeError(
                    "❌ **Invalid Groq API key.** Please check your key in Streamlit secrets.\n"
                    "Get a free key at https://console.groq.com"
                ) from e

            # Rate limit — back off and retry
            if "rate_limit" in err_str or "429" in err_str:
                if attempt < retries:
                    wait = 15 * (attempt + 1)   # 15s, then 30s
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    "⚠️ **Groq rate limit reached.** The free tier has per-minute token limits.\n\n"
                    "**Try:** Wait 30 seconds and re-upload your document.\n"
                    "**Or:** Use a shorter document (remove cover pages, indexes, boilerplate).\n"
                    "**Tip:** The pipeline needs ~40 seconds to complete — avoid clicking twice."
                ) from e

            # Context / token length errors
            if "context_length" in err_str or "too long" in err_str:
                raise RuntimeError(
                    "📄 **Document is too long.** "
                    "Try a shorter document or remove boilerplate pages (cover, index) before uploading."
                ) from e

            # Connection / transient errors — retry with backoff
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue

    raise RuntimeError(
        f"❌ **Groq API failed after {retries + 1} attempts.** "
        f"Error: {last_error}"
    )


# ─────────────────────────────────────────────
# FULL PIPELINE — Agent 0 → 1 → 2 → 3
# ─────────────────────────────────────────────
def analyze_document(doc_text: str, language: str = "English") -> dict:
    """
    Full 4-agent pipeline with doc type detection.

    Model assignment:
      Agent 0, 1 → MODEL_FAST  (llama-3.1-8b-instant,  ~14,400 tok/min free)
      Agent 2, 3 → MODEL_HEAVY (llama-3.3-70b-versatile, ~6,000 tok/min free)

    _INTER_AGENT_DELAY seconds of sleep between each call lets Groq's
    per-minute token counters partially reset, preventing rate-limit errors
    on the free tier without requiring a paid key.

    Returns dict with all agent outputs.
    Raises RuntimeError with a user-friendly message on failure.
    """
    if not doc_text or not doc_text.strip():
        raise RuntimeError(
            "📄 **Document appears to be empty.** "
            "The file may be a scanned image PDF (no embedded text). "
            "Try copying the text manually or use an OCR tool first."
        )

    if len(doc_text.strip()) < 50:
        raise RuntimeError(
            "📄 **Document has too little text to analyse.** "
            "Make sure the PDF contains selectable text, not just images."
        )

    # ── Agent 0 — Detect Document Type ──────────────────────────────
    # Uses MODEL_FAST — only needs 3000 chars of context, 8B handles it fine
    try:
        doc_meta_raw = run_agent(
            DOC_TYPE_PROMPT,
            f"Classify this legal document:\n\n{doc_text[:3000]}",
            model=MODEL_FAST,
        )
        doc_meta = parse_doc_meta(doc_meta_raw)
        doc_type = doc_meta.get("DOC_TYPE", "Other")
    except RuntimeError:
        raise
    except Exception:
        doc_meta_raw = "DOC_TYPE: Other\nPARTIES: Unknown\nJURISDICTION: India (General)\nSUMMARY: Legal document"
        doc_meta = parse_doc_meta(doc_meta_raw)
        doc_type = "Other"

    time.sleep(_INTER_AGENT_DELAY)   # let token counter breathe

    # ── Agent 1 — Extract Clauses (context-aware) ───────────────────
    # Uses MODEL_FAST — structured extraction, no deep reasoning needed
    try:
        clauses_raw = run_agent(
            build_clause_analyzer_prompt(doc_type),
            f"Extract all clauses from this {doc_type}:\n\n{doc_text}",
            model=MODEL_FAST,
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"❌ Clause extraction failed unexpectedly: {e}") from e

    time.sleep(_INTER_AGENT_DELAY)

    # ── Agent 2 — Evaluate Risk (context-aware) ─────────────────────
    # Uses MODEL_HEAVY — needs legal judgment, not just pattern matching
    try:
        risks_raw = run_agent(
            build_risk_evaluator_prompt(doc_type),
            f"Evaluate risk for each clause in this {doc_type}:\n\n{clauses_raw}",
            model=MODEL_HEAVY,
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"❌ Risk evaluation failed unexpectedly: {e}") from e

    time.sleep(_INTER_AGENT_DELAY)

    # ── Agent 3 — Advise + Law Citations + Final Decision ───────────
    # Uses MODEL_HEAVY — Indian law citations + nuanced advice
    try:
        advisor_prompt = build_advisor_prompt(language, doc_type)
        final_raw = run_agent(
            advisor_prompt,
            f"Generate advice, Indian law citations, and final decision for these clauses:\n\n{risks_raw}",
            max_tokens=3000,
            model=MODEL_HEAVY,
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"❌ Legal advice generation failed unexpectedly: {e}") from e

    return {
        "doc_meta":       doc_meta,
        "doc_type":       doc_type,
        "doc_meta_raw":   doc_meta_raw,
        "agent1_clauses": clauses_raw,
        "agent2_risks":   risks_raw,
        "agent3_final":   final_raw,
    }


# ─────────────────────────────────────────────
# NEGOTIATION SCRIPTS (on-demand, Agent 4)
# ─────────────────────────────────────────────
def generate_negotiation_scripts(
    clauses: list,
    doc_type: str,
    language: str = "English",
) -> str:
    """
    Agent 4: Generates negotiation WhatsApp + Email scripts
    for all HIGH and MEDIUM risk clauses.
    Uses MODEL_HEAVY — tone and persuasion quality matters here.
    """
    risky_clauses = [c for c in clauses if c.get("RISK") in ("HIGH", "MEDIUM")]
    if not risky_clauses:
        return "No high or medium risk clauses found — no negotiation needed!"

    clause_summary = ""
    for c in risky_clauses:
        clause_summary += f"CLAUSE: {c.get('CLAUSE', 'Unknown')}\n"
        clause_summary += f"RISK: {c.get('RISK', 'MEDIUM')}\n"
        clause_summary += f"WHY RISKY: {c.get('WHY THIS IS RISKY', '')}\n"
        clause_summary += f"WHAT IT MEANS: {c.get('WHAT IT MEANS', '')}\n===\n"

    try:
        return run_agent(
            build_negotiation_prompt(language, doc_type),
            f"Write negotiation scripts for these risky clauses:\n\n{clause_summary}",
            max_tokens=3000,
            model=MODEL_HEAVY,
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"❌ Negotiation script generation failed: {e}") from e


# ─────────────────────────────────────────────
# SCENARIO SIMULATION
# ─────────────────────────────────────────────
def simulate_scenario(doc_text: str, language: str = "English") -> str:
    lang_instruction = (
        "Respond in simple Hindi."
        if language == "Hindi"
        else "Respond in plain English."
    )
    prompt = f"""You are a legal risk analyst.
The user is about to sign this document WITHOUT negotiating any clauses.
{lang_instruction}

Give a SHORT, honest risk summary (max 200 words). Cover:
1. Top 2 financial risks
2. Key rights lost
3. Single worst-case scenario

Be direct. No filler."""

    try:
        return run_agent(
            prompt,
            f"Risk summary for signing as-is:\n\n{doc_text[:1500]}",
            max_tokens=400,
            model=MODEL_HEAVY,
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"❌ Scenario simulation failed unexpectedly: {e}") from e


# ─────────────────────────────────────────────
# CONSUMER COURT MOTION (Agent 5)
# ─────────────────────────────────────────────
def generate_consumer_motion(
    doc_text: str,
    clauses: list,
    complainant_name: str,
    complainant_address: str,
    opposite_party: str,
    opposite_party_address: str,
    grievance_summary: str,
    relief_amount: str,
    district: str,
    state: str,
    language: str = "English",
) -> str:
    """
    Agent 5: Drafts a Consumer Forum complaint under CPA 2019.
    The complainant can self-represent per Order III CPC.
    Uses MODEL_HEAVY — formal legal drafting requires precision.
    """
    high_clauses = [c for c in clauses if c.get("RISK") == "HIGH"]
    clause_summary = ""
    for c in high_clauses[:5]:
        clause_summary += (
            f"- {c.get('CLAUSE')}: {c.get('WHY THIS IS RISKY', '')} "
            f"[Law: {c.get('INDIAN_LAW', 'CPA 2019')}]\n"
        )

    user_content = f"""
COMPLAINANT: {complainant_name}
ADDRESS: {complainant_address}
DISTRICT/FORUM: {district}, {state}

OPPOSITE PARTY: {opposite_party}
OPPOSITE PARTY ADDRESS: {opposite_party_address}

GRIEVANCE: {grievance_summary}
RELIEF SOUGHT: Rs.{relief_amount} + compensation

HIGH-RISK CLAUSES IDENTIFIED:
{clause_summary or "No specific clauses -- general grievance."}

DOCUMENT EXCERPT (first 1200 chars):
{doc_text[:1200]}
"""

    try:
        return run_agent(
            CONSUMER_MOTION_PROMPT,
            user_content,
            max_tokens=1800,
            model=MODEL_HEAVY,
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"❌ Consumer Court motion draft failed: {e}") from e


# ─────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────
def parse_doc_meta(raw: str) -> dict:
    """Parse Agent 0 output into a dict."""
    meta = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        for key in ["DOC_TYPE", "PARTIES", "JURISDICTION", "SUMMARY"]:
            if line.startswith(f"{key}:"):
                meta[key] = line.replace(f"{key}:", "").strip()
    return meta


def parse_output(raw: str) -> tuple[list[dict], dict]:
    """Parse Agent 3 output into clause list + final decision dict."""
    clauses = []
    final_decision = {}

    if "FINAL_DECISION_START" in raw:
        parts = raw.split("FINAL_DECISION_START")
        clauses_raw = parts[0]
        decision_raw = parts[1].replace("FINAL_DECISION_END", "").strip()

        for line in decision_raw.split("\n"):
            line = line.strip()
            for key in ["OVERALL_RISK", "DANGER_SCORE", "FINAL_DECISION", "REASON", "KEY_CLAUSES_TO_NEGOTIATE"]:
                if line.startswith(f"{key}:"):
                    final_decision[key] = line.replace(f"{key}:", "").strip()
    else:
        clauses_raw = raw

    sections = clauses_raw.split("===")
    for section in sections:
        section = section.strip()
        if not section:
            continue
        clause = {}
        for line in section.split("\n"):
            line = line.strip()
            for key in ["CLAUSE", "RISK", "RISK_SCORE", "WHY THIS IS RISKY", "WHAT IT MEANS", "INDIAN_LAW", "SUGGESTION"]:
                if line.startswith(f"{key}:"):
                    clause[key] = line.replace(f"{key}:", "").strip()
        if "CLAUSE" in clause:
            clauses.append(clause)

    return clauses, final_decision


def parse_negotiation_scripts(raw: str) -> list[dict]:
    """Parse Agent 4 output into a list of negotiation script dicts."""
    scripts = []
    sections = raw.split("===")
    for section in sections:
        section = section.strip()
        if not section:
            continue
        script = {}
        lines = section.split("\n")
        current_key = None
        current_value_lines: list[str] = []

        for line in lines:
            line_stripped = line.strip()
            matched = False
            for key in ["CLAUSE", "WHATSAPP_MESSAGE", "EMAIL_SUBJECT", "EMAIL_BODY"]:
                if line_stripped.startswith(f"{key}:"):
                    if current_key:
                        script[current_key] = "\n".join(current_value_lines).strip()
                    current_key = key
                    current_value_lines = [line_stripped.replace(f"{key}:", "").strip()]
                    matched = True
                    break
            if not matched and current_key:
                current_value_lines.append(line_stripped)

        if current_key:
            script[current_key] = "\n".join(current_value_lines).strip()

        if "CLAUSE" in script:
            scripts.append(script)

    return scripts


def calculate_danger_score(clauses: list, final_decision: dict) -> int:
    """
    Calculate a 0-100 danger score.
    Prefers the LLM-provided DANGER_SCORE if available,
    otherwise computes from clause RISK_SCOREs.
    """
    if "DANGER_SCORE" in final_decision:
        try:
            score = int(final_decision["DANGER_SCORE"])
            return max(0, min(100, score))
        except (ValueError, TypeError):
            pass

    if not clauses:
        return 0

    total, count = 0, 0
    high_count = medium_count = 0

    for c in clauses:
        risk = c.get("RISK", "LOW")
        try:
            score = int(c.get("RISK_SCORE", 0))
            total += score
            count += 1
        except (ValueError, TypeError):
            weight = {"HIGH": 8, "MEDIUM": 4, "LOW": 1}.get(risk, 2)
            total += weight
            count += 1

        if risk == "HIGH":
            high_count += 1
        elif risk == "MEDIUM":
            medium_count += 1

    if count == 0:
        return 0

    base_score = int((total / count) * 10)
    bonus = min(20, high_count * 5 + medium_count * 2)
    return max(0, min(100, base_score + bonus))
