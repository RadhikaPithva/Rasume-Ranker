import streamlit as st
import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import AzureChatOpenAI

# Load Azure environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_name="gpt-4o",
    temperature=0.3
)

# Streamlit config
st.set_page_config("📄 Resume Ranker", layout="centered")
st.title("📄 Resume Ranker with Azure LLM")

st.markdown("Upload a **Job Description** and **resumes** to get AI-based ranking and feedback.")

# JD Upload Options
st.subheader("📌 Job Description")
jd_file = st.file_uploader("Upload JD (PDF)", type=["pdf"])
jd_text_manual = st.text_area("Or paste JD text here (if no PDF)")

# Resume Upload
st.subheader("📎 Resumes")
resumes = st.file_uploader("Upload up to 10 Resume PDFs", type=["pdf"], accept_multiple_files=True)

# LLM Debug Toggle
debug_mode = st.checkbox("Show raw LLM response for debugging", value=False)

# Extract PDF text
import re

def extract_json_block(text):
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    return match.group(0) if match else None

def extract_text(uploaded_file):
    import fitz
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text


def score_resume(jd, resume, show_debug=False):
    prompt = f"""
You are a hiring expert.

Compare the resume to the job description and rate how well it fits (0–100), with a short reason.

Return ONLY valid JSON like:
{{
  "score": 85,
  "reason": "Strong match in skills and role."
}}

Job Description:
{jd[:4000]}

Resume:
{resume[:4000]}
"""
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if show_debug:
            st.text_area("🧠 LLM Raw Output", content, height=100)

        json_text = extract_json_block(content)
        if not json_text:
            return {"score": 0, "reason": "❌ Could not extract JSON from LLM"}
        return json.loads(json_text)

    except json.JSONDecodeError:
        return {"score": 0, "reason": "❌ Invalid JSON from LLM"}
    except Exception as e:
        return {"score": 0, "reason": f"❌ LLM error: {str(e)}"}

# Main logic
if (jd_file or jd_text_manual.strip()) and resumes:
    jd_text = extract_text(jd_file) if jd_file else jd_text_manual.strip()
    results = []

    with st.spinner("Scoring resumes..."):
        for resume in resumes[:10]:
            resume_text = extract_text(resume)
            res = score_resume(jd_text, resume_text, debug_mode)
            results.append({
                "name": resume.name,
                "score": res["score"],
                "reason": res["reason"]
            })

    # Sort and show
    results.sort(key=lambda x: x["score"], reverse=True)
    st.subheader("📊 Ranked Results")

    for r in results:
        st.write(f"**{r['name']}** — Score: {r['score']} ⭐")
        st.caption(r["reason"])

    # Download button
    df = pd.DataFrame(results)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results as CSV", csv, "ranked_resumes.csv", "text/csv")

elif not (jd_file or jd_text_manual.strip()):
    st.info("🔹 Please upload a JD file or paste the JD text.")

elif not resumes:
    st.info("🔹 Please upload at least one resume PDF.")