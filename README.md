# 📄 AI Resume ATS Analyzer

Upload a resume (PDF, DOCX or TXT) and get an **estimated ATS score**, a category
breakdown, missing keywords, prioritised improvements and rewritten bullet points.
Optionally paste a job description to get a targeted keyword match.

Built with **Streamlit** and **Google Gemini (Flash)**.

## Features

- Reads PDF, DOCX and TXT resumes (including text inside DOCX tables)
- Overall ATS score (0-100) with a 5-category breakdown
- Strengths, weaknesses and prioritised, section-by-section fixes
- Before/after rewrites of weak bullet points
- Missing keywords, or a job-description match when a JD is pasted
- Format checklist: contact info, standard sections, length, quantified achievements

## How the score works

No real ATS publishes its scoring rules, so this is an **estimate** that blends
transparent rules with AI judgement:

| Category | Weight | Source |
|---|---|---|
| Format & Structure | 25% | Rule-based: contact details, standard sections, length, share of bullets with numbers |
| Content Quality | 25% | Gemini |
| Keywords & Relevance | 20% | Gemini (matched against the job description if you provide one) |
| Impact & Achievements | 20% | Gemini |
| Clarity & Readability | 10% | Gemini |

The overall score is computed in Python from these category scores, so the maths is
consistent and not left to the model. Use the score as guidance, not a guarantee.

## Project structure

```
resume-ats-analyzer/
├── app.py            # Streamlit app
├── requirements.txt  # Python dependencies
└── README.md
```

## Run locally

1. **Get a free Gemini API key:** https://aistudio.google.com/apikey

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Provide your API key** (pick one):

   - Paste it into the sidebar when the app runs, **or**
   - Create `.streamlit/secrets.toml`:

     ```toml
     GEMINI_API_KEY = "your-key-here"
     ```

   - Or set an environment variable:

     ```bash
     export GEMINI_API_KEY="your-key-here"     # Windows PowerShell: $env:GEMINI_API_KEY="your-key-here"
     ```

4. **Start the app**

   ```bash
   streamlit run app.py
   ```

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | Streamlit secrets, env var, or sidebar | none (required) |
| `GEMINI_MODEL` | Streamlit secrets, env var, or sidebar | `gemini-flash-latest` |

`gemini-flash-latest` is Google's alias for the newest Flash model. Google retires
older model versions over time, so if you ever see a "model not found" error,
change the model name in the sidebar to a current Flash model listed at
https://ai.google.dev/gemini-api/docs/models.

## Deploy on Streamlit Community Cloud

1. Push this project to a **GitHub** repository (never commit your API key).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repository and branch, and set the main file to `app.py`.
4. Open **Advanced settings → Secrets** and add:

   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```

5. Click **Deploy**. Pushing new commits to GitHub redeploys automatically.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Very little text could be extracted" | The PDF is probably a scanned image. Export a text-based PDF or DOCX. ATS software struggles with scans too. |
| "Model not found" | Set the model name in the sidebar to a current Flash model. |
| "Rate limit or quota reached" | Free-tier limits apply. Wait a minute and retry. |
| API key error | Check the key at https://aistudio.google.com/apikey and that it has no extra spaces. |

## Privacy

Resume text is sent to Google's Gemini API for analysis. This app does not store
uploaded files or results. Check Google's API data-use terms for your tier before
uploading sensitive documents.

## Limitations

- The ATS score is an estimate, not the output of any real ATS.
- Multi-column or graphic-heavy PDFs may extract in a jumbled order.
- AI suggestions can be wrong. Review every rewrite and never add experience you don't have.
