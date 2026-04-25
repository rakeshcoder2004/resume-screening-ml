import streamlit as st
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pdfminer.high_level import extract_text

# REPORT IMPORTS
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# ------------------------
# TITLE
# ------------------------

st.title("AI Resume Screening System")
st.write("Upload your resume and find best job role using Machine Learning")

# ------------------------
# LOAD DATASET
# ------------------------

df = pd.read_excel("resume_dataset.xlsx")

# ------------------------
# CLEAN TEXT
# ------------------------

def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

df["Resume"] = df["Resume"].apply(clean_text)

# ------------------------
# TRAIN MODEL
# ------------------------

tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(df["Resume"])
y = df["Category"]

model = LogisticRegression(max_iter=200)
model.fit(X, y)

# ------------------------
# REPORT FUNCTION (UPDATED 🔥)
# ------------------------

def create_pdf(prediction, top_jobs, top_scores, skills, score, missing, advice, strong_area):

    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []

    # TITLE
    content.append(Paragraph("<font size=18 color=blue><b>AI Resume ATS Report</b></font>", styles['Title']))
    content.append(Spacer(1, 15))

    # ATS SCORE
    ats_score = min(score + 10, 100)
    content.append(Paragraph(f"<font color=green><b>ATS Score:</b> {ats_score}/100</font>", styles['Normal']))
    content.append(Spacer(1, 10))

    # PREDICTION
    content.append(Paragraph(f"<b>Predicted Role:</b> {prediction}", styles['Normal']))
    content.append(Spacer(1, 10))

    # TOP JOBS
    content.append(Paragraph("<font color=purple><b>Top Job Matches:</b></font>", styles['Heading2']))
    for job, sc in zip(top_jobs, top_scores):
        content.append(Paragraph(f"{job} - {sc}%", styles['Normal']))
    content.append(Spacer(1, 10))

    # GRAPH
    plt.figure()
    plt.bar(top_jobs, top_scores)
    plt.title("Top Job Matches")

    graph_path = "graph.png"
    plt.savefig(graph_path)
    plt.close()

    content.append(Image(graph_path, width=400, height=250))
    content.append(Spacer(1, 10))

    # SKILLS
    content.append(Paragraph("<font color=orange><b>Skills Found:</b></font>", styles['Heading2']))
    content.append(Paragraph(", ".join(skills), styles['Normal']))
    content.append(Spacer(1, 10))

    # STRONG AREA
    content.append(Paragraph(f"<b>Strong Area:</b> {strong_area}", styles['Normal']))
    content.append(Spacer(1, 10))

    # SCORE
    content.append(Paragraph(f"<b>Resume Score:</b> {score}/100", styles['Normal']))
    content.append(Spacer(1, 10))

    # MISSING
    content.append(Paragraph("<font color=red><b>Recommended Skills:</b></font>", styles['Heading2']))
    content.append(Paragraph(", ".join(missing), styles['Normal']))
    content.append(Spacer(1, 10))

    # ADVICE
    content.append(Paragraph("<b>Career Advice:</b>", styles['Heading2']))
    content.append(Paragraph(advice, styles['Normal']))

    doc.build(content)
    return "report.pdf"

# ------------------------
# SKILL DATABASE
# ------------------------

skills_db = {
"programming":["python","java","c","c++","javascript","sql"],
"ml":["machine learning","deep learning","nlp","tensorflow","pandas","numpy","sklearn"],
"web":["html","css","react","node","bootstrap"],
"cloud":["aws","docker","kubernetes"],
"iot":["iot","arduino","raspberry pi","embedded","microcontroller","sensors"]
}

# ------------------------
# FILE UPLOAD
# ------------------------

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:

    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.read())

    resume_text = extract_text("temp_resume.pdf")
    cleaned_resume = clean_text(resume_text)

    vector = tfidf.transform([cleaned_resume])

    prediction = model.predict(vector)[0]
    probs = model.predict_proba(vector)[0]

    top3_index = np.argsort(probs)[-3:][::-1]

    # OUTPUT
    st.subheader("Predicted Job Role")
    st.success(prediction)

    # TOP JOBS
    st.subheader("Top 3 Suitable Jobs")

    top_jobs = []
    top_scores = []

    for i in top3_index:
        job = y.unique()[i]
        score_val = round(probs[i]*100,2)

        top_jobs.append(job)
        top_scores.append(score_val)

        st.write(job," - ",score_val,"%")

    # GRAPH
    st.subheader("Job Match Graph")

    plt.figure()
    plt.bar(top_jobs, top_scores)
    st.pyplot(plt)

    # SKILLS
    detected_skills = []

    for category in skills_db:
        for skill in skills_db[category]:
            if skill in cleaned_resume:
                detected_skills.append(skill)

    st.subheader("Skills Found in Resume")
    st.write(detected_skills)

    # SKILL COUNT
    st.subheader("Total Skills Found")
    st.write(len(detected_skills))

    # STRONG AREA
    ml_count = prog_count = web_count = iot_count = 0

    for skill in detected_skills:
        if skill in skills_db["ml"]:
            ml_count += 1
        if skill in skills_db["programming"]:
            prog_count += 1
        if skill in skills_db["web"]:
            web_count += 1
        if skill in skills_db["iot"]:
            iot_count += 1

    st.subheader("Strong Area")

    max_area = max(ml_count, prog_count, web_count, iot_count)

    if max_area == ml_count:
        strong_area_text = "Machine Learning"
    elif max_area == prog_count:
        strong_area_text = "Programming"
    elif max_area == web_count:
        strong_area_text = "Web Development"
    else:
        strong_area_text = "IoT"

    st.write(strong_area_text)

    # SCORE
    score = min(len(detected_skills)*8,100)

    st.subheader("Resume Score")
    st.write(str(score) + " / 100")

    # MISSING
    ds_skills = ["python","machine learning","statistics","sql","deep learning"]

    missing = []
    for skill in ds_skills:
        if skill not in detected_skills:
            missing.append(skill)

    st.subheader("Recommended Skills to Improve")
    st.write(missing)

    # ADVICE
    st.subheader("Career Advice")

    if score < 40:
        advice_text = "Add more technical skills to improve resume strength"
    elif score < 70:
        advice_text = "Good profile, improve advanced skills"
    else:
        advice_text = "Strong resume for technical roles"

    st.write(advice_text)

    # DOWNLOAD REPORT
    st.subheader("Download Report")

    pdf_file = create_pdf(
        prediction,
        top_jobs,
        top_scores,
        detected_skills,
        score,
        missing,
        advice_text,
        strong_area_text
    )

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="Download Full Report",
            data=f,
            file_name="resume_report.pdf",
            mime="application/pdf"
        )
