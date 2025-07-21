import streamlit as st
import pdfplumber
import docx2txt
import pandas as pd
import numpy as np
import re
import spacy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load NLP Model
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Sample job dataset
jobs_data = pd.DataFrame({
    "Job Title": ["Data Analyst", "ML Engineer", "Software Developer"],
    "Skills": [
        "SQL, Python, Excel, Tableau",
        "Python, Machine Learning, TensorFlow, Scikit-learn",
        "Java, Python, Git, Web Development"
    ],
    "Location": ["Delhi", "Bangalore", "Remote"]
})

# === Helpers ===
@st.cache_data
def extract_text(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif uploaded_file.name.endswith(".docx"):
        text = docx2txt.process(uploaded_file)
    return text

def extract_keywords(text):
    doc = nlp(text.lower())
    return list(set([
        token.lemma_ for token in doc
        if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop and token.is_alpha
    ]))

def extract_email(text):
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return emails[0] if emails else "Not Found"

def extract_phone(text):
    phones = re.findall(r"\+?\d[\d\s\-]{8,}\d", text)
    return phones[0] if phones else "Not Found"

def extract_experience(text):
    match = re.search(r"(\d+)\+?\s+years", text.lower())
    return match.group(1) if match else "Not Mentioned"

def skill_match(job_skills, resume_skills):
    job_set = set([skill.strip().lower() for skill in job_skills.split(',')])
    resume_set = set(resume_skills)
    matched = job_set & resume_set
    missing = job_set - resume_set
    return matched, missing

# === UI ===
st.set_page_config(page_title="AI Resume Screener", layout="centered")
st.title("🤖 AI Resume Screener & Job Matcher")
st.markdown("Upload your resume, and let AI find your best-fit jobs.")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
location_filter = st.selectbox("🌍 Filter jobs by location", options=["All"] + list(jobs_data["Location"].unique()))

if uploaded_file:
    with st.spinner("🔍 Analyzing your resume..."):
        resume_text = extract_text(uploaded_file)
        resume_keywords = extract_keywords(resume_text)
        resume_embedding = embedder.encode([" ".join(resume_keywords)])

        jobs_data["Job_Embedding"] = jobs_data["Skills"].apply(lambda x: embedder.encode([x])[0])
        jobs_data["Similarity"] = jobs_data["Job_Embedding"].apply(lambda x: cosine_similarity([x], resume_embedding)[0][0])
        
        filtered_jobs = jobs_data if location_filter == "All" else jobs_data[jobs_data["Location"] == location_filter]
        matched = filtered_jobs.sort_values("Similarity", ascending=False).head(3)

        # Extracted info
        email = extract_email(resume_text)
        phone = extract_phone(resume_text)
        experience = extract_experience(resume_text)

    st.success("✅ Resume analyzed successfully!")

    st.subheader("📄 Resume Summary")
    st.markdown(f"**Email:** {email}  \n**Phone:** {phone}  \n**Experience:** {experience} years")
    st.markdown(f"**Extracted Keywords (Top 10):** {', '.join(resume_keywords[:10])}")

    st.subheader("🎯 Top Job Matches")
    for idx, row in matched.iterrows():
        matched_skills, missing_skills = skill_match(row['Skills'], resume_keywords)

        st.markdown(f"""
        ### 🔹 {row['Job Title']}
        **📍 Location:** {row['Location']}  
        **🛠 Required Skills:** {row['Skills']}  
        **✅ Matched Skills:** {', '.join(matched_skills) if matched_skills else 'None'}  
        **❌ Missing Skills:** {', '.join(missing_skills) if missing_skills else 'None'}  
        **📊 Match Score:** {round(row['Similarity'] * 100, 2)}%
        """)

    # Skill match pie chart
    st.subheader("📈 Skill Match Overview")
    labels = ['Matched Skills', 'Missing Skills']
    sizes = [len(matched_skills), len(missing_skills)]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # CSV Download
    st.subheader("⬇️ Download Matched Jobs")
    csv = matched.drop(columns=["Job_Embedding"]).to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "matched_jobs.csv", "text/csv")

