# ⚖️ ClauseIQ — AI Legal Advisor for India

> **Upload any Indian legal document. Get clause-by-clause risk analysis, Indian law citations, a 0–100 Danger Score, ready-to-send negotiation messages, and a Consumer Court complaint draft — in seconds.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Groq](https://img.shields.io/badge/powered%20by-Groq-orange.svg)](https://console.groq.com)

---

## 📌 Project Overview

ClauseIQ is a multi-agent AI application that helps everyday Indians understand the legal documents they are asked to sign — rental agreements, employment contracts, college bonding letters, loan documents, NDAs, and more.

Most people sign documents without understanding them. ClauseIQ changes that by running a 6-agent AI pipeline that extracts every clause, evaluates its risk under Indian law, and tells you exactly what to say to negotiate.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🗂️ Auto Document Detection | Identifies document type (Rental, Employment, Loan, etc.) |
| 🔍 Clause Extraction | Extracts every significant clause from the document |
| ⚠️ Risk Scoring | Rates each clause HIGH / MEDIUM / LOW with a 1–10 risk score |
| ⚖️ Indian Law Citations | Cites exact acts and sections (e.g., Transfer of Property Act 1882) |
| 🧠 Danger Score | Computes a 0–100 document-level danger score |
| 🔏 Sign / Don't Sign | Final AI verdict with reasoning |
| ✍️ Negotiation Scripts | Ready-to-send WhatsApp messages and professional emails |
| 🏛️ Consumer Court Motion | Drafts a CPA 2019 complaint for self-representation |
| 🌐 Hindi Support | Full analysis and scripts available in Hindi |
| 📥 Report Download | Download full analysis as a Markdown report |

---

## 🏗️ Architecture — 6-Agent Pipeline

```
Document Upload
      │
      ▼
┌─────────────┐
│  Agent 0    │  Detects: Rental Agreement / Employment Contract / Loan / NDA / etc.
│  Classifier │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Agent 1    │  Extracts all clauses with context-aware hints per document type
│  Extractor  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Agent 2    │  Scores each clause: HIGH / MEDIUM / LOW + 1–10 risk score
│  Risk Eval  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Agent 3    │  Explains in plain language, cites Indian laws, outputs
│  Legal Adv  │  SIGN / SIGN WITH CHANGES / DO NOT SIGN verdict
└──────┬──────┘
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
┌─────────────┐                      ┌─────────────┐
│  Agent 4    │  WhatsApp + Email    │  Agent 5    │  Consumer Court
│  Negotiator │  negotiation scripts │  CPA Draft  │  motion (CPA 2019)
└─────────────┘                      └─────────────┘
```

**Model:** Groq — `llama-3.3-70b-versatile`  
**Latency:** ~15–30 seconds for the full pipeline (Agents 0–3)

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **AI Backend:** Groq API — Llama 3.3 70B Versatile
- **PDF Parsing:** PyPDF2
- **Language:** Python 3.9+
- **Deployment:** Streamlit Community Cloud

---

## 💻 Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ClauseIQ.git
cd ClauseIQ
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your Groq API key

Create the file `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Get a **free** API key at [console.groq.com](https://console.groq.com).

> **Important:** Never commit `secrets.toml` to Git. It is already in `.gitignore`.

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🚀 Deployment on Streamlit Community Cloud

1. Push your code to a **public GitHub repository**  
   (make sure `.streamlit/secrets.toml` is in `.gitignore`)

2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**

3. Select your GitHub repository and set:
   - **Main file path:** `app.py`

4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

5. Click **Deploy** — your app is live!

---

## 📁 Project Structure

```
ClauseIQ/
├── app.py              # Streamlit frontend
├── agent.py            # Multi-agent backend (Groq + Llama 3.3)
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## 📸 Screenshots

> _Add screenshots here after deployment_

| Upload & Analyse | Clause Analysis | Danger Score | Negotiation Scripts |
|---|---|---|---|
| _(screenshot)_ | _(screenshot)_ | _(screenshot)_ | _(screenshot)_ |

---

## ⚠️ Legal Disclaimer

ClauseIQ is an **AI-powered informational tool only**. It does **not** constitute legal advice, and no attorney-client relationship is created by using this application.

- Law citations are AI-generated and may not reflect the latest legislative amendments.
- Always verify information with a qualified Indian lawyer before signing any document or filing any complaint.
- The Consumer Court motion drafts are starting points only — review and customise before filing.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for everyday Indians who deserve to understand what they sign.*
